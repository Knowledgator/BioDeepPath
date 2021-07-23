import sys
import numpy as np

relation = sys.argv[1]

dataPath = sys.argv[2]

dataPath_ = dataPath + 'tasks/' + relation


ent_id_path = dataPath + '/entities.tsv'
rel_id_path = dataPath + '/relations.tsv'

featurePath = dataPath_ + '/path_to_use.txt'
feature_stats =  dataPath_ + '/path_stats.txt'
test_data_path = dataPath_ + '/test.pairs'

f1 = open(ent_id_path)
f2 = open(rel_id_path)
content1 = f1.readlines()
content2 = f2.readlines()
f1.close()
f2.close()

entity2id = {}
relation2id = {}
for line in content1:
	entity2id[line.split()[1]] = int(line.split()[0])

for line in content2:
	relation2id[line.split()[1]] = int(line.split()[0])


ent_vec = np.loadtxt(dataPath + '/entity2vec.bern')
rel_vec = np.loadtxt(dataPath+ '/relation2vec.bern')

f = open(test_data_path)
test_data = sorted(f.readlines())
f.close()

test_pairs = []
test_labels = []
# queries = set()
for line in test_data:
	e1 = line.split()[0]
	e2 = line.split()[1]

	test_pairs.append((e1,e2))
	label = 1 if line[-2] == '+' else 0
	test_labels.append(label)


aps = []
query = test_pairs[0][0]
y_true = []
y_score = []
query_samples = []

score_all = []

rel = relation
relation_vec = rel_vec[relation2id[rel],:]


for idx, sample in enumerate(test_pairs):
	if sample[0] == query:
		e1_vec = ent_vec[entity2id[sample[0]],:]
		e2_vec = ent_vec[entity2id[sample[1]],:]
		score = -np.sum(np.square(e1_vec + relation_vec - e2_vec))
		score_all.append(score)
		y_score.append(score)
		y_true.append(test_labels[idx])
		query_samples.append(sample)
	else:
		query = sample[0]
		count = zip(y_score, y_true, query_samples)
		count = sorted(count, key = lambda x:x[0], reverse=True)

		ranks = []
		correct = 0
		for idx_, item in enumerate(count):
			if item[1] == 1:
				correct +=  1
				ranks.append(correct/(1.0+idx_))
		if len(ranks)==0:
			ranks.append(0)
		aps.append(np.mean(ranks))
		# if len(aps) % 10 == 0:
			# print 'How many queries:', len(aps)
			# print np.mean(aps)
		y_true = []
		y_score = []
		query_samples = []
		e1_vec = ent_vec[entity2id[sample[0]],:]
		e2_vec = ent_vec[entity2id[sample[1]],:]

		score = -np.sum(np.square(e1_vec + relation_vec - e2_vec))
		score_all.append(score)
		y_score.append(score)
		y_true.append(test_labels[idx])
		query_samples.append(sample)

score_label = zip(score_all, test_labels)
score_label_ranked = sorted(score_label, key = lambda x:x[0], reverse=True)


mean_ap = np.mean(aps)
print('TransE MAP: ', mean_ap)



