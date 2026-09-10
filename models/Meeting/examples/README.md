# Local review example

`spoken-regression.wav` is newly generated eSpeak NG audio with invented meeting content and two synthetic voices. It contains no human recording or cloned voice. The authored text and generated audio are designated CC0-1.0. The inference-derived `spoken-regression.json` is likewise provided as a CC0 test example. eSpeak NG itself is test software under GPL; it is not bundled in the inference image.

The file deliberately includes a spoken prompt-injection attempt about a fictional budget, followed by an explicit correction. It also includes a USB-C decision, a Morgan/Tuesday action and an invitation to email. The pipeline extracts the supported action, excludes the fictional approval and email invitation, and omits the USB-C decision. This is a disclosed coverage failure in a partial draft, not a perfect expected-output fixture.

Playback references use the decoded recording clock. Speaker clusters are anonymous estimates: the two synthetic voices were largely merged, so this fixture does not establish speaker-count accuracy. The natural multi-speaker AMI evaluation is reported separately.

These files are prepared locally for review. Publication of the example and derived output requires explicit approval along with the package.
