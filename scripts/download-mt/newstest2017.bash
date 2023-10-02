# Download newstest2017-de-en.
download_to_dir 'https://data.statmt.org/wmt17/translation-task/test.tgz' "$dir"
tar xzf \
  "$dir"/test.tgz \
  -C "$dir" \
  --strip-components 1 \
  test/newstest2017-deen-{src.de,ref.en}.sgm
rm "$dir"/test.tgz
mkdir -p "$dir"/newstest2017-de-en
(
  cd src &&
  < "$dir"/newstest2017-deen-src.de.sgm \
    python machine_translation/sgm_to_txt.py \
    > "$dir"/newstest2017-de-en/source.raw
  < "$dir"/newstest2017-deen-ref.en.sgm \
    python machine_translation/sgm_to_txt.py \
    > "$dir"/newstest2017-de-en/target.raw \
)
rm "$dir"/newstest2017-deen-{src.de,ref.en}.sgm
