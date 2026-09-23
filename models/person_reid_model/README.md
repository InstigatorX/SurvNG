# Person ReID models

Preferred People-identity / tracking model:

```bash
scripts/install-person-reid-model.sh
```

That writes `osnet_x1_0_msmt17.xml` (torchreid OSNet-x1.0, MSMT17, MIT). SurvNG
applies ImageNet RGB normalization for `osnet_*` files.

Intel Open Model Zoo `person-reidentification-retail-0286` remains the
installer fallback (`scripts/install-docker-models.sh`). It uses raw BGR
input. Body galleries are not interchangeable across these embedding spaces.
