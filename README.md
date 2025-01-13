<div align="center">

<img src="https://raw.githubusercontent.com/ValkyrieSystems/sarkit/main/docs/source/_static/sarkit_logo.png" width=200>

[![Tests](https://github.com/ValkyrieSystems/sarkit/actions/workflows/tests.yml/badge.svg)](https://github.com/ValkyrieSystems/sarkit/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/sarkit/badge/?version=latest)](https://sarkit.readthedocs.io/en/latest/?badge=latest)

</div>

**SARkit** is a suite of Synthetic Aperture Radar (SAR)-related tools in Python developed and maintained by the National Geospatial-Intelligence Agency (NGA) to encourage the use of SAR data standards.

With SARkit, you can:

* read and write SAR standards files (CPHD, SICD, SIDD)
* interact with SAR XML metadata using more convenient Python objects
* check SAR data/metadata files for inconsistencies

## Origins
This project was developed at the National Geospatial-Intelligence Agency (NGA) as the modern successor to
[SarPy](https://github.com/ngageoint/sarpy).

## License
The software use, modification, and distribution rights are stipulated within the MIT license
(See [`LICENSE`](LICENSE) file).

## Contributing and Development
Contributions are welcome; for details see the [contributing guide](./CONTRIBUTING.md).

A few tips for getting started using [PDM](https://pdm-project.org/en/latest/) are below:


```shell
$ pdm install -G:all  # install SARkit with optional & dev dependencies
$ pdm run nox  # run lint and tests
$ pdm run nox -s docs  # build documentation
```
