INPUT = images/
OUTPUT = results/

${OUTPUT}%.csv : ${INPUT}%.jpg
	mkdir -p ${INPUT} ${OUTPUT}${basename ${@F}}
	echo "${INPUT}${basename ${@F}}.jpg"
	echo "${OUTPUT}"
	python3 driver.py ${INPUT}${basename ${@F}}.jpg ${OUTPUT}

ALLFILES=$(subst ${INPUT}, ${OUTPUT}, $(wildcard ${INPUT}*.jpg))

exec: ${ALLFILES:.jpg=.csv}
	touch ${OUTPUT}results.csv
	python3 aggregator.py ${OUTPUT}


clean:
	rm -rf ${OUTPUT}
	rm ${OUTPUT}

.DEFAULT_GOAL := exec


# Current potential downside: while result generation is optimized only for changed/new images, aggregation
# will combine ALL .csv irrespective of repeated information