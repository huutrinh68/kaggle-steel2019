'''return scheduler'''
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_scheduler(optimizer, type, args):
	'''
	return scheduler
	---
	params
		optmizer:
		type: str
			name of scheduler
		args:
	return
		scheduler
	'''
	if type.lower() == 'reducelronplateau':
		scheduler = ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
	return scheduler