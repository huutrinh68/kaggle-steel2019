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
		# scheduler = ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
		scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

	## log
	log.write(f'\n--------------------\n')
	log.write(f'\nscheduler    = {type}\n')
	log.write(f'factor         = {args.factor}\n')
	log.write(f'patience       = {args.patience}\n')

	return scheduler