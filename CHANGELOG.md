# Changelog

## [0.7.0](https://github.com/openfoodfacts/labelr/compare/v0.6.0...v0.7.0) (2025-11-07)


### Features

* improve export & train command ([622dc80](https://github.com/openfoodfacts/labelr/commit/622dc806f89404d5b5a24626ac4ea353f6e54a96))
* improve yolo-train script ([977e4d3](https://github.com/openfoodfacts/labelr/commit/977e4d32605f8633b8add0c23244ad1af7273812))


### Bug Fixes

* create labelr.utils.parse_hf_repo_id func ([ca98d35](https://github.com/openfoodfacts/labelr/commit/ca98d350a30249a0267dfe45a944b5b7a0a03da5))
* fix bug where the wrong object was saved in predictions.parquet ([bc2d883](https://github.com/openfoodfacts/labelr/commit/bc2d883924efd468d2695e357a24a8b3946a0709))
* save predictions.parquet features in labelr library ([e88cc8e](https://github.com/openfoodfacts/labelr/commit/e88cc8e26dccd103d140e3aa4340a78dffa4819b))


### Technical

* add tutorial for object detection ([0cc4d11](https://github.com/openfoodfacts/labelr/commit/0cc4d11ee5542944726fbff696ab81042d6ff202))
* **deps:** add fiftyone as optional dependency ([d1464e2](https://github.com/openfoodfacts/labelr/commit/d1464e28b416c00b93de651d32ce9a9154aef53c))

## [0.6.0](https://github.com/openfoodfacts/labelr/compare/v0.5.0...v0.6.0) (2025-11-06)


### Features

* improve yolo training ([#7](https://github.com/openfoodfacts/labelr/issues/7)) ([00f7ccb](https://github.com/openfoodfacts/labelr/commit/00f7ccb3e7727b2e47faa4e666bb7d94c5e3a865))

## [0.5.0](https://github.com/openfoodfacts/labelr/compare/v0.4.1...v0.5.0) (2025-11-05)


### Features

* add train-yolo package ([#4](https://github.com/openfoodfacts/labelr/issues/4)) ([2d6caec](https://github.com/openfoodfacts/labelr/commit/2d6caec71359f587c57a902f832c459c34b5547b))
* improve train-yolo package ([#6](https://github.com/openfoodfacts/labelr/issues/6)) ([e2ccf67](https://github.com/openfoodfacts/labelr/commit/e2ccf67fa3d6641439bba5ccc79dc17759045ed7))


### Bug Fixes

* increase shared memory size during training ([268b6fa](https://github.com/openfoodfacts/labelr/commit/268b6fa1f88e1bf7892b2736f7ff63ca7b09c9a3))

## [0.4.1](https://github.com/openfoodfacts/labelr/compare/v0.4.0...v0.4.1) (2025-10-31)


### Bug Fixes

* add new command `export_from_ultralytics_to_hf_classification` ([#2](https://github.com/openfoodfacts/labelr/issues/2)) ([ce9cc33](https://github.com/openfoodfacts/labelr/commit/ce9cc336ddb6b8a11536727a7c71e63feb7b5e42))

## [0.4.0](https://github.com/openfoodfacts/labelr/compare/v0.3.0...v0.4.0) (2025-10-31)


### Features

* add a new command to generate a Label Studio config file ([3a50aaa](https://github.com/openfoodfacts/labelr/commit/3a50aaa190cc295065d0626bce93ad54f7a8d95f))
* allow to export from ultralytics to HF for classification tasks ([f6e10d2](https://github.com/openfoodfacts/labelr/commit/f6e10d29e65aa58db9687fc12587b14b72431e6e))
* improve add-split command ([59a36d1](https://github.com/openfoodfacts/labelr/commit/59a36d1ec8ade569a2d6524ea771a2b1fc61a575))
* improve format_object_detection_sample_from_hf function ([f4bd310](https://github.com/openfoodfacts/labelr/commit/f4bd31065b43f1c37c26eb1e22f366d5f7f7406d))
* replace triton backend by Robotoff ([4eca175](https://github.com/openfoodfacts/labelr/commit/4eca175c260910361237e4ea3ba4835b8bcfdd0f))


### Bug Fixes

* add new options to export command ([30e4129](https://github.com/openfoodfacts/labelr/commit/30e412992f5a046637deb812f390d3ccfe0d43ac))
* fix add-split command ([91c7d77](https://github.com/openfoodfacts/labelr/commit/91c7d77eb750c1fa2739dfa9ffabaa9ba367dfac))


### Technical

* add release please ([694704f](https://github.com/openfoodfacts/labelr/commit/694704f8d5f1ac836afc7aa18a0b19ec023f2d49))
* release 0.3.0 ([09d35ea](https://github.com/openfoodfacts/labelr/commit/09d35ea404b587b8f77a6412b51fe26378b19555))
* update package version in lock file ([ac44d81](https://github.com/openfoodfacts/labelr/commit/ac44d81434c649b8fb3f7cdaa0e7e4845c4f145b))
