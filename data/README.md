# Dataset

Photographs of dual-layer QR markers laser-engraved on metallic substrates,
captured with a phone and a macro lens under diffuse illumination and
varying capture conditions. Every image shows the same physical payload, so one
ground-truth map applies to all of them. 

## `raw/<geometry>_<scale>/`

406 photographs, JPEG, grouped by the internal marker geometry (`triangle`,
`corner`, `square`, `cross`) and by the encoder module rendering scale (`s10` or
`s15` pixels per QR module). Filenames come from the capture session
(`MANUAL_*`, `PHOTO_*`, `qr_*_pid1`) and carry no meaning beyond identity.

Each folder also contains the encoder output for that marker —
`qr_final_<geometry>_<scale>.png` and a `_large` rendering. These are the digital
QRs that were not engraved and added as ideal example, the pipeline skips any file whose
name starts with `qr_final_`.

## `warped/<geometry>_<scale>/`

389 rectified grayscale crops, PNG — the output of the localization cascade in
`src/localize_run_all.py` for the images it localized. Each is a view of the Version-4 (33×33) symbol.

The 303 images in the four `*_s15` folders are the corpus used for the
classifier experiments in the paper (88 triangle, 77 square, 74 cross,
64 corner). s15 is equivalent to 0.5 x 0.5 cm in real world.

## `ground_truth_blobs.txt`

337 lines, one `row,col` pair per QR module that carries a hidden bit of value 1
(zero-based, in the 33×33 module grid). All remaining data modules carry 0.

The hidden channel occupies the first 664 non-functional data modules in
row-major order, functional regions (finders, timing, alignment, format
information, quiet zone) are excluded and are never modified by the encoder.

The same map applies to every image, since all photographs show the same
physical payload.

## License

CC BY 4.0. Please cite the paper if you use these images.
