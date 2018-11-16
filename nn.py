import pickle
import torch
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import math
from sklearn.preprocessing import normalize

CHARGES_DICT = {9: 0, 10: 1, 11: 2, 1: 5, 6: 4, 13: 5, 5: 6, 7: 7, 4: 8, 8: 9, 12: 10, 14: 11, 2: 12, 3: 13, 0:14}
ALPHA = 4


def log_to_tb_to_x(loss, n_iter):
	writer.add_scalar("train", loss.data[0], n_iter)

def log_to_tb_to_x_test(loss, n_iter):
	writer.add_scalar("test", loss.data[0], n_iter)

def log_to_tb_to_recid(loss, n_iter):
	writer.add_scalar("train-recid", loss.data[0], n_iter)

def log_to_tb_to_recid_test(loss, n_iter):
	writer.add_scalar("test-recid", loss.data[0], n_iter)

def log_to_tb_to_race(loss, n_iter):
	writer.add_scalar("train-race", loss.data[0], n_iter)

def log_to_tb_to_race_test(loss, n_iter):
	writer.add_scalar("test-race", loss.data[0], n_iter)

def log_line(loss_type, loss, n_iter):
	writer.add_scalar(loss_type, loss.data[0], n_iter)

def log_to_tb(loss, loss_recid, loss_violence, loss_failure, is_test, n_iter):
	prefix = "train"
	if is_test:
		prefix = "test"
	log_line(prefix + '/loss', loss, n_iter)
	log_line(prefix + '/loss_recid', loss_recid, n_iter)
	log_line(prefix + '/loss_violence', loss_violence, n_iter)
	log_line(prefix + '/loss_failure', loss_failure, n_iter)

train_file = open('train.pkl', 'rb')
test_file = open('test.pkl', 'rb')
val_file = open('valid.pkl', 'rb')
misd_fel_file = open('misd_fel.pkl', 'rb')

train = pickle.load(train_file)
test = pickle.load(test_file)
val = pickle.load(val_file)
misd_fel = pickle.load(misd_fel_file)

print(len(train))
print(len(test))
print(len(val))

train_file.close()
test_file.close()
misd_fel_file.close()

X_train, Y_train = [], []
X_test, Y_test = [], []
race_train, race_test = [], []

logs_path = 'runs/model_1H_264_RECID_NOADVERSARYAPRIL27c4_prison_days'

writer = SummaryWriter(logs_path)

print("model_1H_264_RECID_NOADVERSARYAPRIL27c4_prison_days")

model_name = './model_1H_264_RECID_NOADVERSARYAPRIL27c4_prison_days.pth'
adversary_name = './ADVERSARY.model_1H_264_RECID_NOADVERSARYAPRIL27c4_prison_days.pth'


def get_temp(value, misd_fel):
	temp = []
	race = [0, 0, 0, 0, 0, 0]
	race[value['race']] = 1
	custody = [0, 0, 0, 0, 0, 0]
	custody[value['custody_status']] = 1
	marriage = [0, 0, 0, 0, 0, 0, 0]
	marriage[value['marital_status']] = 1
	temp.append(value['sex'])
	temp.extend(race)
	temp.append(value['juv_misd'])
	temp.append(value['juv_felony'])
	temp.append(value['juv_other'])
	temp.append(misd_fel['prison_num'])
	temp.extend(custody)
	temp.append(misd_fel['jail_days'])
	temp.append(value['num_charges_before_compas'])
	temp.append(misd_fel['num_misdemeanors'])
	temp.append(misd_fel['jail_num'])
	temp.append(value['priors_count'])
	temp.append(misd_fel['num_felonies'])
	temp.append(CHARGES_DICT[value['c_charge_degree']])
	temp.append(value['age'])
	temp.extend(marriage)
	# temp.append(misd_fel['prison_days'])
	return temp

for key, value in train.items():
	temp = get_temp(value, misd_fel[key])
	# For adversary, only have as training data if black or white
	if (value['race'] != 0 and value['race'] != 3) or value['is_recid'] < 0 or value['compas_recid_decile'] == -1 or value['compas_failure_decile'] == -1 or value['compas_violence_decile'] == -1:
		pass
	else:
		X_train.append(temp)
		# Uncomment for one prediction instead
		Y_train.append(value['is_recid'])
		# Y_train.append(value['compas_recid'])
		# Y_train.append([value['compas_recid'], value['compas_violence'], value['compas_failure']])
		# Hack: can /3 to get 0/1 labels because black = 0, white = 3
		race_train.append(value['race']/3)

for key, value in test.items():
	temp = get_temp(value, misd_fel[key])
	# For adversary, only have as training data if black or white
	if (value['race'] != 0 and value['race'] != 3) or value['is_recid'] < 0 or value['compas_recid_decile'] == -1 or value['compas_failure_decile'] == -1 or value['compas_violence_decile'] == -1:
		pass
	else:
		X_test.append(temp)
		# Uncomment for one prediction instead
		Y_test.append(value['is_recid'])
		# Y_test.append(value['compas_recid'])
		# Y_test.append([value['compas_recid'], value['compas_violence'], value['compas_failure']])
		# Hack: can /3 to get 0/1 labels because black = 0, white = 3
		race_test.append(value['race']/3)


N, D_in, H, H_adversary, D_out, D_out_adversary = len(X_train), len(X_train[0]), 264, 100, 1, 1

# For recid prediction
X_train, Y_train, race_train = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(np.expand_dims(np.asarray(Y_train), 1))), Variable(torch.Tensor(race_train))
X_test, Y_test, race_test = Variable(torch.Tensor(X_test)), Variable(torch.Tensor(np.expand_dims(np.asarray(Y_test), 1))), Variable(torch.Tensor(race_test))

# X_train, Y_train, race_train = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(Y_train)), Variable(torch.Tensor(race_train))
# X_test, Y_test, race_test = Variable(torch.Tensor(X_test)), Variable(torch.Tensor(Y_test)), Variable(torch.Tensor(race_test))
keep_prob = .5
learning_rate = 1e-4
num_epochs = 20000

# COMPAS prediction model
# Optimal with 2-4 layers
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Dropout(keep_prob),
    # torch.nn.Linear(H, H),
    # torch.nn.ReLU(),
    # torch.nn.Dropout(keep_prob),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)

# Adversary to predict race
adversary = torch.nn.Sequential(
    torch.nn.Linear(D_out, H_adversary),
    torch.nn.ReLU(),
    torch.nn.Dropout(keep_prob),
    torch.nn.Linear(H_adversary, D_out_adversary),
    torch.nn.Sigmoid(),
)

# loss_fn = torch.nn.MSELoss(size_average=True)
loss_fn = torch.nn.BCELoss(size_average=True)


loss_fn_adversary = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_adversary = torch.optim.Adam(adversary.parameters(), lr=learning_rate)


for t in range(num_epochs):
	y_pred = model(X_train)
	# race_input = torch.cat((y_pred, Y_train), 1)
	
	race_pred = adversary(y_pred)
	race_loss = loss_fn_adversary(race_pred, race_train)

	loss = loss_fn(y_pred, Y_train)
	# loss_recid = loss_fn(y_pred[:, 0], Y_train[:, 0])
	# loss_violence = loss_fn(y_pred[:, 1], Y_train[:, 1])
	# loss_failure = loss_fn(y_pred[:, 2], Y_train[:, 2])
	# loss = loss_recid + loss_violence + loss_failure

	y_pred_test = model(X_test)
	# race_input_test = torch.cat((y_pred_test, Y_test), 1)
	race_pred_test = adversary(y_pred_test)
	race_loss_test = loss_fn_adversary(race_pred_test, race_test)

	loss_test = loss_fn(y_pred_test, Y_test)
	# loss_recid_test = loss_fn(y_pred_test[:, 0], Y_test[:, 0])
	# loss_violence_test = loss_fn(y_pred_test[:, 1], Y_test[:, 1])
	# loss_failure_test = loss_fn(y_pred_test[:, 2], Y_test[:, 2])
	# loss_test = loss_recid_test + loss_violence_test + loss_failure_test

	if t % 100 == 0:
		log_to_tb_to_race(race_loss, t)
		log_to_tb_to_race_test(race_loss_test, t)
		log_to_tb_to_recid(loss, t)
		log_to_tb_to_recid_test(loss_test, t)
		# log_line("train/loss_recid", loss, t)
		# log_line("test/loss_recid", loss_test, t)
		# log_to_tb(loss, loss_recid, loss_violence, loss_failure, False, t)
		# log_to_tb(loss_test, loss_recid_test, loss_violence_test, loss_failure_test, True, t)
	if t % 10000 == 0:
		torch.save(model, model_name)
		torch.save(adversary, adversary_name)

	# Backprop race_loss
	# race_loss.backward(retain_graph=True)
	# Zero generator
	# optimizer.zero_grad()
	# Step adversary
	# optimizer_adversary.step()
	# Zero adversary (unnecessary?)
	# optimizer_adversary.zero_grad()
	# Calculate loss to backprop generator
	# loss = loss - ALPHA*race_loss
	# Backprop loss
	loss.backward()
	# Step generator
	optimizer.step()
	# Zero gradients for generator and adversary
	optimizer.zero_grad()
	# optimizer_adversary.zero_grad()

writer.close()

model.eval()

y_pred = model(X_test)
loss = loss_fn(y_pred, Y_test)
print("Loss", loss)



