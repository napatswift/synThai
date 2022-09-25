import os

import numpy as np
from PIL import Image

from synthtiger import components, layers, templates

class SynThaiger(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.corpus = components.BaseCorpus(**config.get("corpus", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.color = components.RGB(**config.get("color", {}))
        self.bgcolor = components.RGB(**config.get("bgcolor", {}))
        self.layout = components.FlowLayout(**config.get("layout", {}))
        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {}),
        )
        self.texture = components.Switch(
            components.BaseTexture(), **config.get("texture", {})
        )

    def generate(self):
        number_of_sample = np.random.randint(0, 4)
        texts = [self.corpus.data(self.corpus.sample()) for _ in range(number_of_sample)]
        font = self.font.sample()
        colors = [self.color.data(self.color.sample()) for _ in range(number_of_sample)]

        text_group = layers.Group([layers.TextLayer(text, color=color, **font) for text, color in zip(texts, colors)])
        self.layout.apply(text_group)

        bg_layer = layers.RectLayer(text_group.size, self.bgcolor.data(self.bgcolor.sample()))
        bg_layer.topleft = text_group.topleft
        self.texture.apply([bg_layer])
        image = (text_group + bg_layer).output()
        
        image = self._postprocess_image(image)

        text = " ".join(texts)

        data = {
            "image": image,
            "label": text,
            "font": font,
        }

        return data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        gt_path = os.path.join(root, "gt.txt")
        self.gt_file = open(gt_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        font = data["font"]

        h, w, d = image.shape

        shard = str(idx // 1000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        image_path = os.path.join(root, image_key)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_path, quality=np.random.randint(30, 95))

        self.gt_file.write(f"{image_key}\t{label}\n")

    def end_save(self, root):
        self.gt_file.close()

    def _postprocess_image(self, image):
        layer = layers.Layer(image)
        self.postprocess.apply([layer])
        out = layer.output()
        return out
