
% clear

Cs = load('SVM_Cs.txt','-ascii')
% 

%% 
subject='A'
%subject='AandB'

num_channels = [2, 4, 8, 16, 32, 64];
num_chosen_channels = load(sprintf('%s_num_chosen_channels.txt',subject),'-ascii')

accuracy = [];
for ch = num_channels
    filename = sprintf('%s_training_accuracy_SVM_ch%d.txt', subject, ch)
    acc = load(filename, '-ascii');
    accuracy = [accuracy acc];
end
[max_accuracy, max_idx] = max(accuracy);
Cs(max_idx)



figure()
subplot(121)
plot(log10(Cs),accuracy)
xlabel('log_{10}C')
ylabel('accuracy')
box off
title(subject)

subplot(122)
plot(max_accuracy, 'o--')
xlabel('number of channels')
ylabel('max accuracy')

set(gca,'xtick', 1:length(max_accuracy))
%set(gca,'xticklabel', num_channels)
set(gca,'xticklabel', num_chosen_channels)
box off


%% 
num_chosen_channels_AandB = num_chosen_channels;
max_accuracy_AandB = max_accuracy;

%% 
num_chosen_channels_A = num_chosen_channels;
max_accuracy_A = max_accuracy;

%%
close all
plot(log2(num_chosen_channels_A), max_accuracy_A, 's--'); hold on
plot(log2(num_chosen_channels_AandB), max_accuracy_AandB, '*:')

legend('A', 'A and B')
xlabel('log_2(number of channels')
ylabel('accuracy')


