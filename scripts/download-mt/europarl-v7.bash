# Download europarl-v7-de-en.
download_to_dir 'https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz' "$dir"
tar xzf \
  "$dir"/training-parallel-europarl-v7.tgz \
  -C "$dir" \
  --strip-components 1 \
  training/europarl-v7.de-en.{de,en}
rm "$dir"/training-parallel-europarl-v7.tgz
mkdir -p "$dir"/europarl-v7-de-en
mv "$dir"/europarl-v7.de-en.de "$dir"/europarl-v7-de-en/source.raw
mv "$dir"/europarl-v7.de-en.en "$dir"/europarl-v7-de-en/target.raw
