# Create a small toy dataset for testing.
toy_train_size=10000
toy_valid_size=100
toy_test_size=100
mkdir -p "$dir"/toy-{train,valid,test}
for side in source target; do
  head -"$toy_train_size" "$dir"/europarl-v7-de-en/"$side".raw > "$dir"/toy-train/"$side".raw
  head -"$toy_valid_size" "$dir"/newstest2016-de-en/"$side".raw > "$dir"/toy-valid/"$side".raw
  head -"$toy_test_size" "$dir"/newstest2017-de-en/"$side".raw > "$dir"/toy-test/"$side".raw
done
