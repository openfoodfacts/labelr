# Changelog

## [0.10.0](https://github.com/openfoodfacts/labelr/compare/v0.9.0...v0.10.0) (2026-01-13)


### Features

* add first draft of train-unsloth package ([19998dc](https://github.com/openfoodfacts/labelr/commit/19998dc373fc1502d77d94fb210f6dacd2386045))
* add labelr datasets export-llm-ds CLI command ([276c5b6](https://github.com/openfoodfacts/labelr/commit/276c5b6f10fe5a2955085a95e4148b860188b4d9))
* **train-unsloth:** add a validate command ([9c837ea](https://github.com/openfoodfacts/labelr/commit/9c837ea9d4fd324c6afe672aa897313bb0f9af70))
* **train-unsloth:** add new CLI params ([0112d73](https://github.com/openfoodfacts/labelr/commit/0112d73ef12bbf608c68e65eefa1663879d86a70))
* **train-unsloth:** allow to specify image max size and max seq len ([3de960d](https://github.com/openfoodfacts/labelr/commit/3de960d2baa17c977e0747401f797016616c4c15))
* **train-unsloth:** report to wandb ([0e5fd7c](https://github.com/openfoodfacts/labelr/commit/0e5fd7c8c8321cd54c7ad4589f6f0b1220388244))
* **train-unsloth:** use vLLM to run on validation set ([ecaf560](https://github.com/openfoodfacts/labelr/commit/ecaf56063e88df5826c2a73ec4d0ad9e4490b344))


### Bug Fixes

* create and reuse storage.Client ([d2973b8](https://github.com/openfoodfacts/labelr/commit/d2973b8eed91465d08eac6471d2e4834e605b59f))
* fix import and dependency issues ([15c62ef](https://github.com/openfoodfacts/labelr/commit/15c62eff44ef349357ee992bac3938138c661146))
* fix overwritten imported func ([2377f1b](https://github.com/openfoodfacts/labelr/commit/2377f1b2b7f6e765a5c9519f399c87ff93ea7843))
* import unsloth before trl ([f569fe0](https://github.com/openfoodfacts/labelr/commit/f569fe0069b34e82c425845cb5ebe1c1c0d8779f))
* **train-unsloth:** add new parameters ([dc41867](https://github.com/openfoodfacts/labelr/commit/dc41867396fc3cad0d60f50465463ccb34974177))
* **train-unsloth:** decrease logging-steps default value to 1 ([c4c8f07](https://github.com/openfoodfacts/labelr/commit/c4c8f07c334cce2632ad1312265ac23ec6216e5f))
* **train-unsloth:** don't push the processor ([c3902a5](https://github.com/openfoodfacts/labelr/commit/c3902a5840b62c913caec692a9ff3e597d216134))
* **train-unsloth:** fix bugs in validate command ([3757847](https://github.com/openfoodfacts/labelr/commit/3757847f6a3ef468e36c8a0a0548f79c89176def))
* **train-unsloth:** fix call to upload_file ([05bd156](https://github.com/openfoodfacts/labelr/commit/05bd156bcaded544b2ae036f7492d8ecbb1392c5))
* **train-unsloth:** fix issues with training script ([ea45e37](https://github.com/openfoodfacts/labelr/commit/ea45e37a36fc79bcf845a44f1d66f2d84b5fcce8))
* **train-unsloth:** fix max_length ([6c0e2ac](https://github.com/openfoodfacts/labelr/commit/6c0e2ac8e6d7bbbd91f056a9de5f22299d4b8b41))
* **train-unsloth:** fix Multiprocessing issue ([b88dabb](https://github.com/openfoodfacts/labelr/commit/b88dabb589e5edc08f3433ca7e585241ab146755))
* **train-unsloth:** fix mypy typing issues ([2b24979](https://github.com/openfoodfacts/labelr/commit/2b249798e0fd3233da01dcc443d6410331249ea8))
* **train-unsloth:** fix run_on_validation_set ([308a027](https://github.com/openfoodfacts/labelr/commit/308a0279d8641beece0e4e80f9f25c82b2942616))
* **train-unsloth:** fix run_on_validation_set ([2e9878c](https://github.com/openfoodfacts/labelr/commit/2e9878ccec37c8aa4d450fb47194e1d83c23c11d))
* **train-unsloth:** fix validation run ([881eedb](https://github.com/openfoodfacts/labelr/commit/881eedbe52952206b3bf1261013b77a1888f1dfe))
* **train-unsloth:** free CUDA memory before running validation ([31647ef](https://github.com/openfoodfacts/labelr/commit/31647efee0a9528b2ca8253a5264c0951419c9c0))
* **train-unsloth:** import torch ([985dd1d](https://github.com/openfoodfacts/labelr/commit/985dd1dd826bf77b37ff8b751c0c9f3af54b8972))
* **train-unsloth:** let user authenticate to HF through HF_TOKEN envvar ([72f7175](https://github.com/openfoodfacts/labelr/commit/72f717516b482025db10031eace8f6811d45165b))
* **train-unsloth:** rename tokenizer in processor ([86af6a4](https://github.com/openfoodfacts/labelr/commit/86af6a41441ea2d6751cefce6f27946ffa00b5da))


### Technical

* add labelr export CLI subcommand ([4525fdd](https://github.com/openfoodfacts/labelr/commit/4525fddc8bcbea2b44ea3239d400ca84f980f670))
* create labelr.export subpackage ([57b46b7](https://github.com/openfoodfacts/labelr/commit/57b46b7b7b96b5ed89d10f0a58079d2b674c2baa))
* **deps:** add missing typing libs ([279bbf8](https://github.com/openfoodfacts/labelr/commit/279bbf8070aefacf068caf72bc234c504551a3dd))
* **deps:** add qwen-vl-utils to dependencies ([b417f34](https://github.com/openfoodfacts/labelr/commit/b417f34f32cb71568d0593875c9d2b264538ece0))
* **train-unsloth:** add wandb dep ([e042d85](https://github.com/openfoodfacts/labelr/commit/e042d8564fcaf6c5f1d8934b634364098a3bf148))
* **train-unsloth:** create train-unsloth package ([b789358](https://github.com/openfoodfacts/labelr/commit/b78935833960543a4581e206d04160964934123d))

## [0.9.0](https://github.com/openfoodfacts/labelr/compare/v0.8.0...v0.9.0) (2026-01-05)


### Features

* add commands to launch Gemini batch jobs ([e9bba33](https://github.com/openfoodfacts/labelr/commit/e9bba33e099a7614037ef7b9f0dd99f4f073c204))
* add upload_training_dataset_from_predictions CLI command ([6f2d934](https://github.com/openfoodfacts/labelr/commit/6f2d934b5bf9da702bfb29ae1ea03861b883dfdb))


### Technical

* bump default python version to 3.11 ([cc871a4](https://github.com/openfoodfacts/labelr/commit/cc871a4e17c512390bfc931cbae2983f8cd39b18))
* remove legacy file ([7e05a45](https://github.com/openfoodfacts/labelr/commit/7e05a453c08470499db237da93a09928b31fffc2))

## [0.8.0](https://github.com/openfoodfacts/labelr/compare/v0.7.0...v0.8.0) (2025-11-21)


### Features

* add command to show HF sample ([1ec8e24](https://github.com/openfoodfacts/labelr/commit/1ec8e24a3ecc23f9fd3bf83e9fc7be837eee54b1))
* add module to visualize predictions using fiftyone ([2503ce8](https://github.com/openfoodfacts/labelr/commit/2503ce87962acb9175585c7cfc5f8625d200177d))
* **train-yolo:** allow yolo12 models ([d3c6f47](https://github.com/openfoodfacts/labelr/commit/d3c6f478cb5235f009c1ab2695ebbe46dc8872e6))


### Bug Fixes

* **evaluate:** make --dataset-name optional ([89a8640](https://github.com/openfoodfacts/labelr/commit/89a86408f2f36baf550db37e8baffad99cac13f7))
* **export:** fix issue when exporting ds from LS to HF ([a01e591](https://github.com/openfoodfacts/labelr/commit/a01e59183cda84df8710176792b58c0814ec8eed))
* fix batch size for validation ([a821d9b](https://github.com/openfoodfacts/labelr/commit/a821d9b1acde2c301c6d92eb1aef97dfed92c015))
* minor update on command help text ([46fc1ad](https://github.com/openfoodfacts/labelr/commit/46fc1ade40dc6b96cec725978eec5fbde12f42fb))
* **object-detection:** don't export models with NMS and pin opset ([0c67c53](https://github.com/openfoodfacts/labelr/commit/0c67c53d380b8cea43fa1b7dbd854caf68bf5064))
* **object-detection:** don't export to TensorRT ([8ca9558](https://github.com/openfoodfacts/labelr/commit/8ca95584436341d71ce0b7d2f5dc47d9a1db9743))
* rename some functions ([f36488c](https://github.com/openfoodfacts/labelr/commit/f36488caca6673a0a46405c81ec1d55f3cadc6c0))
* rotate image wrt EXIF Orientation when generating hf sample ([8646459](https://github.com/openfoodfacts/labelr/commit/86464590304826ca9c40c1045715d98e8cb56a73))
* **train-yolo:** run validation using best model for pytorch ([d434ae1](https://github.com/openfoodfacts/labelr/commit/d434ae1c63a06f4cbfc470c08d79720b9195d960))
* **train:** automatically add datestamp at the end of run name ([b5fa5e3](https://github.com/openfoodfacts/labelr/commit/b5fa5e3292529e5f2787a8965b791388c898e5d1))


### Technical

* create a labelr.evaluate.object_detection module ([2a50a1b](https://github.com/openfoodfacts/labelr/commit/2a50a1bb797c0200fdf2e23e0f244468a170cee6))
* **deps:** add uv.lock ([90627ad](https://github.com/openfoodfacts/labelr/commit/90627adfe968f70d429e18f206dfedc69679e7c0))
* improve docstring in projects app ([6be09e3](https://github.com/openfoodfacts/labelr/commit/6be09e34688622e42552c083b8ac36205a1aa2da))
* speedup CLI loading ([8d1f51f](https://github.com/openfoodfacts/labelr/commit/8d1f51ff8a09e114b76a1ad65daa924f1cd39674))
* **train-yolo:** replace definitions and funcs by labelr ([0d34879](https://github.com/openfoodfacts/labelr/commit/0d348791fca759dd29aaf7855c2c791ad6da3a52))

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
