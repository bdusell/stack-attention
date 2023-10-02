# Download newstest2016-de-en.
download_to_dir 'https://data.statmt.org/wmt17/translation-task/dev.tgz' "$dir"
tar xzf \
  "$dir"/dev.tgz \
  -C "$dir" \
  --strip-components 1 \
  dev/newstest2016-deen-{src.de,ref.en}.sgm
rm "$dir"/dev.tgz
mkdir -p "$dir"/newstest2016-de-en
(
  cd src &&
  < "$dir"/newstest2016-deen-src.de.sgm \
    python machine_translation/sgm_to_txt.py \
    > "$dir"/newstest2016-de-en/source.raw
  < "$dir"/newstest2016-deen-ref.en.sgm \
    python machine_translation/sgm_to_txt.py \
    > "$dir"/newstest2016-de-en/target.raw \
)
rm "$dir"/newstest2016-deen-{src.de,ref.en}.sgm
