import segmentation_models_pytorch as smp

def init_network(args, log):
    if args.arch == 'unet':
        model = smp.Unet(
            encoder_name=args.model, 
            encoder_weights=args.encoder_weights, 
            classes=args.classes, 
            activation=args.activation)
    elif args.arch == 'fpn':
        model = smp.FPN(
            encoder_name=args.model, 
            encoder_weights=args.encoder_weights, 
            classes=args.classes, 
            activation=args.activation)
    model = model.to(args.device)

    ## log
    log.write(f'\n--------------------\n')
    log.write(f'model arch         = {args.arch}\n')
    log.write(f'model name         = {args.model}\n')
    log.write(f'encoder_weights    = {args.encoder_weights}\n')
    log.write(f'classes            = {args.classes}\n')
    log.write(f'activation         = {args.activation}\n')
    log.write(f'move to            = {args.device}\n')

    return model