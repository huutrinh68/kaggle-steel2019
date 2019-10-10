'''return scheduler'''
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_scheduler(optimizer, type, args, log):
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

	## log
	log.write(f'--------------------')
	log.write(f'\nscheduler    = {type}\n')
	log.write(f'factor         = {args.factor}\n')
	log.write(f'patience       = {args.patience}\n')

	return scheduler