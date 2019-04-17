import os
import sys
import time
import random

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from game.flappybird import GameState


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def train(model, start):
    replay_memory = []
    game_state = GameState()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # Initial Action
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat(
        (image_data, image_data, image_data, image_data)).unsqueeze(0)

    iteration = 0
    epsilon = model.initial_epsilon
    epsilon_decrements = np.linspace(
        model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    while iteration < model.number_of_iterations:

        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        # if random_action:
        #     print("Random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        # Next state and reward
        image_data_next, reward, terminal = game_state.frame_step(action)
        image_data_next = resize_and_bgr2gray(image_data_next)
        image_data_next = image_to_tensor(image_data_next)
        state_next = torch.cat(
            (state.squeeze(0)[1:, :, :], image_data_next)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(
            np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save (transition) to replay memory
        replay_memory.append((state, action, reward, state_next, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(
            len(replay_memory), model.minibatch_size))

        # unpack
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_next_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_next_batch = state_next_batch.cuda()

        output_next_batch = model(state_next_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_next_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_next
        state = state_next
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model/trained_model_" +
                       str(iteration) + ".pth")

            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon,
                  "action:", action_index.cpu().detach().numpy(),
                  "reward:", reward.numpy()[0][0],
                  "Q max:", np.max(output.cpu().detach().numpy()))


def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat(
        (image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
            # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_next, reward, terminal = game_state.frame_step(action)
        image_data_next = resize_and_bgr2gray(image_data_next)
        image_data_next = image_to_tensor(image_data_next)
        state_next = torch.cat(
            (state.squeeze(0)[1:, :, :], image_data_next)).unsqueeze(0)

        # set state to be state_next
        state = state_next


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/trained_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:
            model = model.cuda()
        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)

if __name__ == "__main__":
    main(sys.argv[1])
