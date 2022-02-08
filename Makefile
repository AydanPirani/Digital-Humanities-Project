INPUT = images/
OUTPUT = results/

results/data/%.csv : images/%.jpg
	mkdir -p ${INPUT} ${OUTPUT}/data ${OUTPUT}/imgs
	echo "Main!"
	echo "${INPUT}${basename ${@F}}.jpg"
	python3 driver.py ${INPUT}${basename ${@F}}.jpg ${OUTPUT}

ALLFILES=$(subst images/, results/data/, $(wildcard images/*.jpg))

exec: ${ALLFILES:.jpg=.csv}
	touch ${OUTPUT}results.csv
	python3 aggregator.py ${OUTPUT}


clean:
	rm -rf ${OUTPUT}/data/*.csv
	rm ${OUTPUT}/results.csv

.DEFAULT_GOAL := exec


# Current potential downside: while result generation is optimized only for changed/new images, aggregation
# will combine ALL .csv irrespective of repeated information