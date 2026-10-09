# exege

Tools for understanding and evaluating scientific machine-learning models: what an
emulator holds inside, sparse autoencoders trained on it, figures, and a local app over
them. The name comes from the Greek stem *exēgē-*, associated with explanation and
interpretation.

> **Research tool.** `exege` is early. What is described here works; expect it to change.

```console
$ pip install exege            # everything but torch
$ pip install 'exege[nn]'      # and torch, for training sparse autoencoders
$ exege --help
```

This is the full install. It holds no code of its own: it installs
[exege-core](https://pypi.org/project/exege-core/) with every extra but `nn`, and both
provide the same `import exege` and the same `exege` command. For the light install,
Click only, with each subpackage's dependencies added as you need them:

```console
$ pip install 'exege-core[latents]'
```

- Documentation: <https://exege.readthedocs.io>
- Source and issues: <https://github.com/mahf708/exege>

MIT.
