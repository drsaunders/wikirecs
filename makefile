SAMPLESIZE=1000000
USERS_PER_CHUNK=500
EARLIEST_TIMESTAMP=2020-05-28T02:59:44Z

edit_histories$(suffix)_0.feather: sampled_users$(suffix).csv
	python pull_edit_histories.py --sampled_users_file  sampled_users$(suffix).csv \
		--edit_histories_file_pattern "edit_histories$(suffix)_{}.feather" \
		--users_per_chunk $(USERS_PER_CHUNK) \
		--earliest_timestamp $(EARLIEST_TIMESTAMP) \

sampled_users$(suffix).csv:
	python get_sample_of_users.py --edit_lookback $(SAMPLESIZE) --outfile="sampled_users$(suffix).csv"


clean:
	rm sampled_users$(suffix).csv
	rm edit_histories$(suffix)_*.feather