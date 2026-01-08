# Changelog

## [1.2.1](https://github.com/t0/rfmux/compare/v1.2.0...v1.2.1) (2026-01-08)


### Bug Fixes

* _auto_bias_kids: mock CRS calls are now asynchronous and need to be awaited ([cdfc5a5](https://github.com/t0/rfmux/commit/cdfc5a5119b67fc34ef0f4ccb57edcdaafcf3b11))
* mock UDP streamer: don't reset sequence numbers when decimation changes ([5007e83](https://github.com/t0/rfmux/commit/5007e8331cfa7ec92f94e5ad91d0071c2eafd012))
* remove duplicate (identical) MockCRS._auto_bias_kids ([36cb46d](https://github.com/t0/rfmux/commit/36cb46d3ea44440a81e00c6a6413b150d3cde637))

## 1.2.0 (2026-01-06)


### Features

* **manual-trigger-release-please:** manual trigger ([0317de3](https://github.com/t0/rfmux/commit/0317de34acc70798867c757c35a80675a4949e83))
* **periscope:** unified tabbed interface with dockable panels ([#44](https://github.com/t0/rfmux/issues/44)) ([7cb76c7](https://github.com/t0/rfmux/commit/7cb76c7b9a0b46f6aa4b617916ce313f710497bb))
* **release-please:** yml file for release please testing ([fa160c8](https://github.com/t0/rfmux/commit/fa160c88971833ec7879a9cba247daebbfb7c7eb))


### Bug Fixes

* **change-test-name:** change test name ([4f774cf](https://github.com/t0/rfmux/commit/4f774cfab8859a79e559467335eb883c39885b97))
* **pfb_psd_mag_bin-error:** fixed the error caused because of an if statement, also adjusted the position of hover labels ([#46](https://github.com/t0/rfmux/issues/46)) ([2073506](https://github.com/t0/rfmux/commit/207350688bbfd53b997c8db3664ce7e254667ec8))
* **py-get-samples-and-update-noise-plot:** when reference is absolute the value is already in volts, it was converting again, its removed now ([#37](https://github.com/t0/rfmux/issues/37)) ([e19bee3](https://github.com/t0/rfmux/commit/e19bee3734442845261138161eb8e43c16bf0130))
* **test-push:** test push ([ca0836b](https://github.com/t0/rfmux/commit/ca0836bde6ce54cbc69be0cf57c600e49e2b67d6))


### Documentation

* **In-the-documents-folder:** Added overall description of the repository ([#28](https://github.com/t0/rfmux/issues/28)) ([cc89540](https://github.com/t0/rfmux/commit/cc895404499763e7a899108beed215b5ddde547f))
