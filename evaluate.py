def run_evaluation(model, train_dataset, valid_dataset, test_dataset):
    for name, dataset in [("Train", train_dataset), ("Validation", valid_dataset), ("Test", test_dataset)]:
        print(f'\n------------- On {name} Set --------------------------\n')
        res = model.evaluate(dataset, batch_size=16, return_dict=True)
        print('________________________')
        print('Loss:           | {:.2f} |'.format(res['loss'] * 100))
        print('Dice Coef:      | {:.2f} |'.format(res['final_output_dice_coef'] * 100))
        print('IOU:            | {:.2f} |'.format(res['final_output_iou'] * 100))
        print('________________________')

