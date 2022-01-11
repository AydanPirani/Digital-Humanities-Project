results/data/%.csv : images/%.jpg
	python3 driver.py ${@F}

ALLFILES=$(subst images/, results/data/, $(wildcard images/*.jpg))

exec: ${ALLFILES:.jpg=.csv}
	python3 aggregator.py

clean:
	rm -rf ./results/data/*.csv
	rm ./results/results.csv

.DEFAULT_GOAL := exec


# Current potential downside: while result generation is optimized only for changed/new images, aggregation
# will combine ALL .csv irrespective of repeated information