import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from opacus.accountants.utils import get_noise_multiplier
from epsilon_calculation import sub_epsilon
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    
    parser.add_argument('--dp-a', action='store_true', help='whether to add noise to the first stage')
    parser.add_argument('--dp-b', action='store_true', help='whether to add noise to the second stage')
    parser.add_argument('--max-grad-norm-a', type=float, default=1.0, help='parameter for differential privacy')
    parser.add_argument('--max-grad-norm-b', type=float, default=1.0, help='parameter for differential privacy')
    parser.add_argument('--epsilon', type=float, default=10., help='parameter for differential privacy')
    parser.add_argument('--delta', type=float, default=1e-5, help='parameter for differential privacy')

    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.iteration] # The list of iterations when we evaluate models and record results.
    eval_it_pool = [20, 50, 100, 200, 500, 1000]
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    # sample_rate_a = args.batch_real * num_classes / len(dst_train)
    # dp_steps_a = int(len(dst_train) / (args.batch_real * num_classes) * args.iteration * args.outer_loop)
    sample_rate_a = args.batch_real / len(dst_train)
    dp_steps_a = args.iteration * args.outer_loop * num_classes
    sample_rate_b = args.batch_real / len(dst_train)
    dp_steps_b = args.iteration * args.outer_loop

    print('DP steps A: ', dp_steps_a)
    print('DP steps B: ', dp_steps_b)

    args.sigma_a = get_noise_multiplier(target_epsilon=args.epsilon, target_delta=args.delta,
                        sample_rate=sample_rate_a,
                        steps=dp_steps_a)
    print('Noise sigma A: ', args.sigma_a)
    args.sigma_b = get_noise_multiplier(target_epsilon=args.epsilon, target_delta=args.delta,
                        sample_rate=sample_rate_b,
                        steps=dp_steps_b)
        
    print('Noise sigma B: ', args.sigma_b)

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)



        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        for it in range(args.iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.



            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0

            ''' update synthetic data '''
            losses = []
            if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                clean_loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    if args.dp_a or args.dp_b:
                        if args.dp_a:
                            output_syn = []
                            for sample_img in img_syn:
                                sample_out = embed(sample_img.unsqueeze(0))
                                clip_coef = min(1, args.max_grad_norm_a / (sample_out.data.norm(2) + 1e-7))
                                sample_out.mul_(clip_coef)
                            output_syn.append(sample_out)
                            output_syn = torch.cat(output_syn)
                        else:
                            output_syn = embed(img_syn)
                        
                        output_real = []
                        for sample_img in img_real:
                            sample_out = embed(sample_img.unsqueeze(0)).detach()
                            
                            if args.dp_a:
                                # print(sample_out.data.norm(2))
                                clip_coef = min(1, args.max_grad_norm_a / (sample_out.data.norm(2) + 1e-7))
                                sample_out.mul_(clip_coef)
                            output_real.append(sample_out.detach())

                        if args.dp_b:
                            for sample_output_real in output_real:
                                losses.append(torch.sum((torch.mean(sample_output_real, dim=0) - torch.mean(output_syn, dim=0))**2))
                        else:
                            output_real = torch.cat(output_real).sum(0)
                            clean_loss += torch.sum((output_real / args.batch_real - torch.mean(output_syn, dim=0))**2)

                            noise = torch.randn_like(output_real) * args.sigma_a * args.max_grad_norm_a
                            output_real = (output_real + noise) / args.batch_real
                            loss += torch.sum((output_real - torch.mean(output_syn, dim=0))**2)
                    
                    else:
                        output_real = embed(img_real).detach()
                        output_syn = embed(img_syn)

                        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            else: # for ConvNetBN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

            if args.dp_b:
                sum_gradient = 0
                batch_size = args.batch_real * num_classes
                for i in range(batch_size):
                    optimizer_img.zero_grad()
                    # Backward pass for each sample
                    losses[i].backward(retain_graph=True if i < batch_size - 1 else False)
                    
                    # Storing gradients
                    gradient = image_syn.grad.clone()
                    clip_coef = min(1, args.max_grad_norm_b / (gradient.data.norm(2) + 1e-7))
                    gradient.mul_(clip_coef)
                    
                    sum_gradient += gradient
                # image_syn.grad.zero_()
                # noise = torch.randn_like(sum_gradient) * args.sigma_b * args.max_grad_norm_b
                # image_syn.grad.add_((sum_gradient + noise)/args.batch_real)
                # optimizer_img.step()
                # loss_avg += torch.Tensor(losses).sum().item()

                
                    
                noise = torch.randn_like(sum_gradient) * args.sigma_b * args.max_grad_norm_b
                # print(sum_gradient.norm(2), noise.norm(2))
                image_syn.grad.zero_()
                image_syn.grad.add_((sum_gradient + noise*10)/args.batch_real)
                clean_grad = sum_gradient/args.batch_real
                noisy_grad = image_syn.grad.clone()
                print(clean_grad.norm(2), noisy_grad.norm(2))
                print((clean_grad-noisy_grad).pow(2).sum() / clean_grad.pow(2).sum())
                print((clean_grad-noisy_grad).pow(2).sum() / noisy_grad.pow(2).sum())
                print()
                optimizer_img.step()
                loss_avg += torch.Tensor(losses).sum().item()
            else:
                optimizer_img.zero_grad()
                clean_loss.backward(retain_graph=True)
                clean_grad = image_syn.grad.clone()

                optimizer_img.zero_grad()
                loss.backward()
                noisy_grad = image_syn.grad.clone()
                optimizer_img.step()
                loss_avg += loss.item()
                print(clean_grad.norm(2), noisy_grad.norm(2))
                print((clean_grad-noisy_grad).pow(2).sum() / clean_grad.pow(2).sum())
                print((clean_grad-noisy_grad).pow(2).sum() / noisy_grad.pow(2).sum())
                print()

                # optimizer_img.zero_grad()
                # loss.backward()
                # optimizer_img.step()
                # loss_avg += loss.item()


            loss_avg /= (num_classes)

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


