%.csv : %.jpg
# 	echo Running Input:  $< output: $@
# 	echo Filename: $(@F)
# 	sleep 1
	python3 driver.py $(@F)

# 	wc $< > $@

test:
	echo "test running!"

ALLFILES=$(wildcard images/*.jpg)


exec: $(ALLFILES:.jpg=.csv)

data: exec
	python3 aggregator.py