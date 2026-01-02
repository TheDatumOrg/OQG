
# ALL_DS=(
#   uqv sald1m space1V LLAMA imageNet bigann netflix CCNEWS deep1m lendb
#   cifar nuswide ARXIV IQUIQUE astro1m audio MNIST geofon ukbench sun
#   NEIC millionSong seismic1m AGNEWS YAHOO CELEBA glove LANDMARK GOOGLEQA
#   texttoimage OBST2024 sift notre tiny5m crawl instancegm CODESEARCHNET gist
# )

ALL_DS=(netflix)


for ds in "${ALL_DS[@]}"; do
  echo "Running dataset: $ds"
  taskset -c 0-127 python train.py --ds "$ds"
  taskset -c 0 python test.py --ds "$ds"
done