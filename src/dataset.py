import torch
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from glob import glob
from torchvision.transforms.v2 import Compose
from tokenizer import LaTeXTokenizer


class LaTeXDataset(Dataset):
    def __init__(
        self,
        inkml_dir: str,
        tokenizer: LaTeXTokenizer = None,
        transform: Compose = None,
    ):
        super().__init__()

        self.inkml_dir = inkml_dir
        self.transform = transform
        self.tokenizer = tokenizer

        self.data = self._create_dataset()

    def _normalize_stroke(
        self, stroke: list, min_x: int, min_y: int, padding: int
    ) -> list:
        normalized = []
        for x, y in stroke:
            new_x = (x - min_x) + padding
            new_y = (y - min_y) + padding
            normalized.append((new_x, new_y))

        return normalized

    def _parse_inkml(self, inkml_path: str, padding=10, line_width=2) -> tuple:
        tree = ET.parse(inkml_path)
        root = tree.getroot()

        namespace = {"ink": "http://www.w3.org/2003/InkML"}

        label = root.find('.//ink:annotation[@type="normalizedLabel"]', namespace)
        label_text = label.text if label is not None else ""

        strokes = []
        for trace in root.findall(".//ink:trace", namespace):
            points = trace.text.strip().split(",")
            stroke = []
            for point in points:
                coords = point.strip().split()
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    stroke.append((x, y))
            if stroke:
                strokes.append(stroke)

        if not strokes:
            return torch.zeros(1, 1, 1), label_text, 1, 1

        all_points = [point for stroke in strokes for point in stroke]
        all_x = [p[0] for p in all_points]
        all_y = [p[1] for p in all_points]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        content_width = max_x - min_x
        content_height = max_y - min_y

        img_width = int(content_width + 2 * padding)
        img_height = int(content_height + 2 * padding)

        normalized_strokes = [
            self._normalize_stroke(s, min_x, min_y, padding) for s in strokes
        ]

        img = Image.new("L", (img_width, img_height), color=255)
        draw = ImageDraw.Draw(img)

        for stroke in normalized_strokes:
            if len(stroke) > 1:
                draw.line(stroke, fill=0, width=line_width)
            elif len(stroke) == 1:
                x, y = stroke[0]
                r = line_width // 2
                draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

        img_array = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_array).unsqueeze(0)

        return tensor, label_text, img_width, img_height

    def _create_dataset(self) -> list:
        files = glob(f"{self.inkml_dir}/*.inkml")
        return files

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        file_path = self.data[idx]

        img, label, _, _ = self._parse_inkml(file_path)

        if self.transform:
            img = self.transform(img)

        if self.tokenizer:
            label = self.tokenizer.encode(label)
            label = torch.tensor(label, dtype=torch.long)

        return img, label
