# Local review example

`spoken-regression.wav` is newly generated eSpeak NG audio with invented meeting content and two synthetic voices. It contains no human recording or cloned voice. The authored text and generated audio are designated CC0-1.0. The inference-derived `spoken-regression.json` is likewise provided as a CC0 test example. eSpeak NG itself is test software under GPL; it is not bundled in the inference image.

The file deliberately includes a spoken prompt-injection attempt about a fictional budget, followed by an explicit correction. It also includes a USB-C decision, a Morgan/Tuesday action and an invitation to email. The pipeline extracts the supported action, excludes the fictional approval and email invitation, and omits the USB-C decision. This is a disclosed coverage failure in a partial draft, not a perfect expected-output fixture.

Playback references use the decoded recording clock. Speaker clusters are anonymous estimates: the two synthetic voices were largely merged, so this fixture does not establish speaker-count accuracy. The natural multi-speaker AMI evaluation is reported separately.

These files are prepared locally for review. Publication of the example and derived output requires explicit approval along with the package.

The authored spoken reference, timing, generator version and audio SHA are in `spoken-reference.json`. ASR preserved R9700, 32 GB, 64 compute units, GFX1201, BF16 and 8-bit weights with normalized spelling, but rendered “AI PRO” as “iPro.” This is a small inspectable technical-vocabulary check, not a general names/numbers accuracy rate.

The compiled-ASR option with native vLLM also has an [actual Studio output](spoken-vllm-source-runtime.json). It retains the USB-C decision, Morgan/Tuesday action and undecided launch issue, each with the supporting transcript segment; it excludes the injected fictional budget and email invitation. This run took 164.09 seconds including sequential stage startup on the 38.90-second fixture. The earlier Transformers output is preserved above; these single first-use runs are not a controlled speed comparison.

The recommended stock-ASR default has a separate [actual Studio output](spoken-stock-default.json): USB-C decision, Morgan/Tuesday action and undecided launch, each with a supporting transcript reference. It rejects the fictional approval and email invitation. The browser completed import, processing, renaming, playback seeking and export without page errors; the exported backend is `stock`. This short workflow took 144.68 seconds including startup. Use the repeated full-recording report for performance comparisons.
