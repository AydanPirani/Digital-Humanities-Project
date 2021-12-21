%.csv : %.jpg
# 	echo Running Input:  $< output: $@
# 	echo Filename: $(@F)
# 	sleep 1
	python3 python/driver.py $(@F)
# 	wc $< > $@


ALLFILES=$(wildcard images/*.jpg)
dummy: $(ALLFILES:.jpg=.csv)

clean: