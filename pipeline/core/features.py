import torch


class _LayerTap:
    def __init__(self, target, layer_ids):
        self.layer_ids = list(layer_ids)
        self._buf = {}
        self._handles = []
        self.enabled = False
        layers = target.model.language_model.layers
        for lid in self.layer_ids:
            h = layers[lid].register_forward_hook(self._make_hook(lid))
            self._handles.append(h)

    def _make_hook(self, lid):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            hs = output[0] if isinstance(output, tuple) else output
            self._buf[lid] = hs

        return hook

    def gather(self):
        return torch.cat([self._buf[lid] for lid in self.layer_ids], dim=-1)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
