clear all;close all;clc;

%% Add biosig toolbox for loading data, modify path as needed. Biosig can 
% be downloaded for free from the Mathworks website
addpath(genpath('./biosig/'));

%% Useful init
load('chanlocs16.mat');
SamplingRate = 512; % Hz
NChEEG = 16;
ChannelLabels = {'Fz','FC3','FC1','FCz','FC2','FC4','C3','C1','Cz','C2','C4',...
    'CP3','CP1','CPz','CP2','CP4'};

%% Butterworth filter
[b, a] = butter(4, [1 10]/(SamplingRate/2), 'bandpass');

%% Load Laplacian matrix for
lap = load('laplacian16.mat');
lap16 = lap.lap; clear lap;

%% Find all BDF data
DataPath = ''; % Set your own path to the downloaded data
GDFfiles = dir([DataPath '/*.gdf']);
GDFfiles = {GDFfiles.name};


%% Load data, extract trials and labels
Trials = [];
Labels = [];
for f=1:length(GDFfiles)

    try
    % Load file with biosig's sload
    [data, header] = sload([DataPath '/' GDFfiles{f}]);
    catch
        continue;
    end
    
    %% Keep trigger channel (useless for this data)
    trigger = data(:,end);
    
    %% Keep EEG channels
    data = data(:,[1:NChEEG]);
    
    %% Events are in header.EVENT, 3 is a Correct trial, 4 is an Error trial
    IndStartTrial = find(ismember(header.EVENT.TYP,[3 4]));
    Labels = [Labels ; header.EVENT.TYP(IndStartTrial)-2]; % Keep labels and remap labels to 1 (Correct) and 2 (Error)
    
    %% Trial extraction
    for tr=1:length(IndStartTrial)    
        Trials = cat(3,Trials,data(header.EVENT.POS(IndStartTrial(tr))-1*SamplingRate:...
            header.EVENT.POS(IndStartTrial(tr))-1,:));
    end
    
end

Trials = permute(Trials, [3 1 2]); % Make trials the first dimension

%% Pre-processing (per trial)
for tr=1:size(Trials,1)
    
    %% Artifact removal (you can try FORCe from the corresponding coding exercise, 
    % however, it takes long and this data is not particularly noisy
    
    %% Spatial filtering
    %% CAR -- DOES IT HELP?
    %Trials(tr,:,:) = car(squeeze(Trials(tr,:,:)));
 
     %% Laplacian -- DOES IT HELP?
     %Trials(tr,:,:) = laplacianSP(squeeze(Trials(tr,:,:)),lap16);
    
    
    %% Spectral filtering
    
    % Apply band-pass filter in [1, 10] Hz
    Trials(tr,:,:) = filtfilt(b,a,squeeze(Trials(tr,:,:)));

    %% Baseline
    % DC removal -- IS IT NECESSARY GIVEN the 0-baselining?
    %Trials(tr,:,:) = removeDC(squeeze(Trials(tr,:,:)));
    
    %% Baseline, remove potential at t=0
    Trials(tr,:,:) = squeeze(Trials(tr,:,:)) - repmat(squeeze(Trials(tr,1,:))',size(Trials,2),1);
    
end

%% Grand averages for each channel
for ch=1:NChEEG
    GACorrect(ch,:) = mean(Trials(Labels==1,:,ch),1);
    GAError(ch,:) = mean(Trials(Labels==2,:,ch),1);
    subplot(4,4,ch);plot([1:size(Trials,2)]/SamplingRate,squeeze(GACorrect(ch,:)),'b');
    hold on;
    subplot(4,4,ch);plot([1:size(Trials,2)]/SamplingRate,squeeze(GAError(ch,:)),'r');
    subplot(4,4,ch);plot([1:size(Trials,2)]/SamplingRate,squeeze(GAError(ch,:)-GACorrect(ch,:)),'k');
    hold off;
    legend({'Correct','Error','Difference'});
    title(ChannelLabels{ch});
    xlabel('Time [s]');
    ylabel('Potential [uV]');
end

%% Topoplots
load('chanlocs16.mat');
% Find min and max of FCz
[maxVal, maxInd] = max(GAError(4,:)-GACorrect(4,:)); 
[minVal, minInd] = min(GAError(4,:)-GACorrect(4,:));
figure(2);
subplot(2,1,1);topoplot(squeeze(GAError(:,minInd)-GACorrect(:,minInd)),chanlocs16,'maplimits',[-4 3]);colorbar;
subplot(2,1,2);topoplot(squeeze(GAError(:,maxInd)-GACorrect(:,maxInd)),chanlocs16,'maplimits',[-4 3]);colorbar;

%% Downsample all trials, check if the ErrP shape is captured adequately
for tr=1:size(Trials)
    DTrials(tr,:,:) = downsample(Trials(tr,:,:),32);
end


%% Feature selection    
% Feature ranking with Fisher Score
FS = fisherscore(reshape(Trials,[size(Trials,1) ...
    size(Trials,2)*size(Trials,3)]),Labels); 
FSMat = reshape(FS,size(Trials,2),size(Trials,3))'; 

% Feature ranking with R2
R2 = rsquared(reshape(Trials,[size(Trials,1) ...
    size(Trials,2)*size(Trials,3)]),Labels); 
R2Mat = reshape(R2,size(Trials,2),size(Trials,3))'; 

% % Feature ranking with CVA -- Use only on downsampled data, takes too
% long
% CVA = cva(reshape(DTrials,[size(DTrials,1) ...
%     size(DTrials,2)*size(DTrials,3)]),Labels); 
% CVAMat = reshape(CVA,size(DTrials,2),size(DTrials,3))'; 


figure(3);
subplot(2,1,1);imagesc(FSMat);colorbar;
set(gca,'XTick',[1 128 256 384 512]);
set(gca,'XTicklabel',[0 0.25 0.5 0.75 1]);
set(gca,'YTick',[1:NChEEG]);
set(gca,'YTicklabel',ChannelLabels);
subplot(2,1,2);imagesc(R2Mat);colorbar;
set(gca,'XTick',[1 128 256 384 512]);
set(gca,'XTicklabel',[0 0.25 0.5 0.75 1]);
set(gca,'YTick',[1:NChEEG]);
set(gca,'YTicklabel',ChannelLabels);


%% Remove Fz as it is susceptible to artifacts
DTrials(:,:,1) = [];

% Leave-one-out (LOO) cross-validation 
NFeatures = 5;
for tr=1:size(DTrials,1) 
    IndTrain = setdiff([1:size(DTrials,1)],tr);
    
    %% Feature selection + classification
    % Perform feature selection and train classifier on training samples (trials)
    R2 = fisherscore(reshape(DTrials(IndTrain,:),[length(IndTrain) ...
    size(DTrials,2)*size(DTrials,3)]), Labels(IndTrain)); 
    R2(isnan(R2)) = 0;
    [~, sortind] = sort(R2,'descend');
    
    TrainSet = DTrials(IndTrain,:);
    TrainSet = TrainSet(:,sortind(1:NFeatures));
    
    TestSample = DTrials(tr,:);
    TestSample = TestSample(sortind(1:NFeatures));
    
    LDAModel = fitcdiscr(TrainSet,Labels(IndTrain),'DiscrimType','quadratic');
    LDAPrediction(tr) = predict(LDAModel,TestSample);
    
    SVMModel = fitcsvm(TrainSet,Labels(IndTrain),'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
    SVMPrediction(tr) = predict(SVMModel,TestSample);
    
    %% Dimensionality reduction + classification
    TrainSet = DTrials(IndTrain,:);
    MeanTrainSet = mean(TrainSet); % I will need this since PCA centers the data
    [PCs,TrainSetPCA,EigenValues] = pca(TrainSet);
    
    %% Sanity check: just confirming that TrainSetPCA is simply: Centered TrainSet x PCs (which it is, indeed)
    %MyTrainSetPCA = (TrainSet - repmat(MeanTrainSet,size(TrainSet,1),1))*PCs;
    % Center TestSet (TestSaple in this case) and project it with PCA
    PCATestSample = (DTrials(tr,:) - MeanTrainSet)*PCs(:,1:NFeatures);
    
    %% Train classifiers with 10 first PCs
    PCA_LDAModel = fitcdiscr(TrainSetPCA(:,1:NFeatures),Labels(IndTrain),'DiscrimType','quadratic'); % Play with linear/quadratic
    PCA_LDAPrediction(tr) = predict(PCA_LDAModel,PCATestSample);
    PCA_SVMModel = fitcsvm(TrainSetPCA(:,1:NFeatures),Labels(IndTrain),'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
    PCA_SVMPrediction(tr) = predict(PCA_SVMModel,PCATestSample);
    
end

% Calculate accuracy (% of correct classifier predictions)
LDAAccuracy = 100*sum(LDAPrediction' == Labels)/length(Labels);
SVMAccuracy = 100*sum(SVMPrediction' == Labels)/length(Labels);
PCA_LDAAccuracy = 100*sum(PCA_LDAPrediction' == Labels)/length(Labels);
PCA_SVMAccuracy = 100*sum(PCA_SVMPrediction' == Labels)/length(Labels);


%% You can extend the results to a confusion matrix (i.e., analytically find the type of errors
%% committed by the classifier)