cd data/
# Prepare refcocog data
if [ ! -d datasets ]; then
  mkdir datasets
  wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
  wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
  wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
  unzip refcocog.zip -d ./datasets/
  unzip refcoco.zip -d ./datasets/
  unzip refcoco+.zip -d ./datasets/
  rm refcocog.zip
  rm refcoco.zip
  rm refcoco+.zip
fi

# Prepare glove data
if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
  unzip glove.840B.300d.zip
  rm glove.840B.300d.zip
if
