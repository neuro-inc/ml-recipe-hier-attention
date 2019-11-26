[ ! -f ../data/imdb_and_vecros.zip ] && wget http://data.neu.ro/imdb_and_vecros.zip -O ../data/imdb_and_vecros.zip
[ ! -d ../data/coco ] && unzip ../data/imdb_and_vecros.zip -d ../data
rm -rf ../data/imdb_and_vecros.zip ../data/__MACOSX