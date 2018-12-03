import uuid
import simplejson as json

from toolz.dicttoolz import valmap


class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(o.value, **self.kwargs)
            return "@@%s@@" % (key,)
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        for k, v in self._replacement_map.items():
            result = result.replace('"@@%s@@"' % (k,), v)
        return result


def __wrap_to_prevent_indentation(obj):
    return NoIndent(obj)


def pretty_json(item):
    if 'feature_maps' in item:
        item['feature_maps'] = valmap(__wrap_to_prevent_indentation, item['feature_maps'])
    if 'activations' in item:
        item['activations'] = valmap(__wrap_to_prevent_indentation, item['activations'])
    return json.dumps(item, indent=4, cls=NoIndentEncoder)
