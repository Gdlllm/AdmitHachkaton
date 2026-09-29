# InstantHMR neural-weight attribution

The bundled `instanthmr.onnx` is the unmodified public InstantHMR checkpoint
published by `momolesang`. Its pinned model card describes distillation from
Meta’s SAM 3D Body; the source repository describes training differently. We do
not independently certify the training history. Weight provenance is pinned
separately from the JavaScript/decoder source licenses.

- [Pinned model card](https://huggingface.co/momolesang/InstantHMR/blob/3504446fc31e7f76fdb1cd7e463189e7cf0fdd0f/README.md)
- [Exact weight download](https://huggingface.co/momolesang/InstantHMR/resolve/3504446fc31e7f76fdb1cd7e463189e7cf0fdd0f/instanthmr.onnx)
- Revision: `3504446fc31e7f76fdb1cd7e463189e7cf0fdd0f`
- Bytes: `80961622`
- SHA-256: `f717558094f57d7c9cd084d7981874dbcc3f08e7308501eb98c12844e2cce3b7`

The pinned Hugging Face model card declares `sam-license` for these weights and
links Meta's SAM 3D Body license. That Hugging Face revision contains no separate
LICENSE file. An unmodified copy of the linked SAM license is included here as
`LICENSE-SAM-WEIGHTS.txt`; the model card is preserved as
`INSTANTHMR-MODEL-CARD.md`.

The copied license is pinned to
[Meta's LICENSE at 4e33ae485b2ea0ecc38328fa39b16b75dd3e79c0](https://github.com/facebookresearch/sam-3d-body/blob/4e33ae485b2ea0ecc38328fa39b16b75dd3e79c0/LICENSE).
Its source URL and checksum are recorded in `weights-license-manifest.json`.
This notice does not replace or relicense that agreement.

The InstantHMR implementation/keypoint regressor is separately Apache-2.0;
MHR and Momentum notices and license texts are alongside these files. Local
changes are browser preprocessing, ONNX execution, bounded alignment and a
JavaScript MHR decoder. The neural weight file itself has not been modified or
retrained.
