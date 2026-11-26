# License and provenance

- Inkling 2 Mini weights, drafter weights, tokenizer and chat template: Thinking Machines, Apache-2.0, repository
  `thinkingmachines/Inkling-2-Mini`, revision `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`. The upstream repository
  states the license in its model card and ships no license file, so `MODEL_LICENSE` holds the Apache-2.0 text.
  `USAGE_POLICY` links to the publisher's acceptable use policy.
- The weights are downloaded into the user's cache folder at first start and checked against
  `checkpoint.lock.json`. The images do not contain them, and this repository publishes no converted copy.
- 3-bit expert files for the super fast image (`inkling-2-mini-w3-rdna4-v1.2.0`): made by Paiton from the expert
  weights of the revision above and published as `EliovpAI/Inkling-2-Mini-W3-INT3-Paiton-RDNA4` at revision
  `0b824fcf035916564fbdd9962acf5177d0ed332a` under Apache-2.0, as a
  modified copy of those weights. [TBD licence check: wording for the derived files, the model card's notice.]
- DFlash2 drafter (`--drafter dflash2`, the launcher's default; `--drafter dspark` serves with the drafter in the
  model checkpoint): weights by incoai, Apache-2.0, repository `incoai/Inkling-2-Mini-DFlash2`, revision
  `4f9088ea66c21f2b99ed657d1941c306b03e84d2`. That repository states the license in its model card and ships no
  license file. The launcher downloads its two files (1.0 GB) into the user's cache folder at the first start with
  this drafter and checks them against `checkpoint.lock.json`. The image does not contain them, and this repository
  publishes no copy. Paiton's library for this drafter was written from the reference implementation in
  vLLM 0.32 (Apache-2.0), read as a specification.
- In the default image the expert weights run in the formats they were trained and published in (NVFP4 and MXFP8).
  The LM head is stored as 8-bit rows when the model loads; everything else stays bf16. The super fast image runs the
  3-bit expert files above and stores the attention projections as 8-bit rows when the model loads.
- The model code in `paiton_vllm_plugin/inkling2` started from the vLLM project's Inkling 2 model code
  (Apache-2.0, copyright the vLLM contributors) and was changed to run on vLLM 0.32 and on Paiton's native
  libraries. The file headers keep the original notices.
- The image contains Paiton's compiled libraries for gfx1201 and gfx1200 (experts, attention, operators, forward sequencer, vision tower, the two drafters).
  The compiler source is private and not part of the image.
- Base image: `rocm/vllm@sha256:380efeed0d5fa65fc044253adc1ce53a04907d1202b30fb36e175370ee39a003` (ROCm 10.2,
  vLLM 0.32). Its notices and license texts stay in place, and the third-party notices of this repository apply.
