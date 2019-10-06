from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_scheduler(optimizer, type, args):
    if type.lower() == 'reducelronplateau':
        ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
