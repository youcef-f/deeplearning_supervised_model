# deep learning4j

**projet de machine learning  supervisé**



le Gradient Descent : permet de minimiser la **fonction cout** dans les algothirme de **deeplearning supervisé**. C'est de cette machine que l'algorithme apprend.

**Learning Rate: ( Alpha )**: défiit la vitesse de convergeance de l'algorithme pour atteindre le minimal.
plus la valeur du learnig rate est elevé , on risque dans ce cas de faire des bascules des valeurs positive et négative de la courbe sans jamais l'atteindre. On oscille autour du minimuim sans jamais l'atteindre.   Plus le learning rate est trop faible on risque alors de convergé vers le minimuim trop lentement et de ne pas atteindre le minimuin rapidement.

![](doc/images/desentGradient.jpg)
![](doc/images/desentGradient1.jpg)



Model  MutliLayer Percetron 
https://www.codeflow.site/fr/article/deeplearning4j
https://www.inceptive.tech/predire-lallure-dune-personne-avec-dl4j/
https://github.com/eugenp/tutorials/blob/master/deeplearning4j/src/main/java/com/baeldung/deeplearning4j/IrisClassifier.java





Notez que la dépendance nd4j-native-platform fait partie des différentes implémentations disponibles.

Il repose sur des bibliothèques natives disponibles pour de nombreuses plates-formes différentes (macOS, Windows, Linux, Android, etc.). Nous pourrions également basculer le backend en nd4j-cuda-8.0-platform , si nous voulions exécuter des calculs sur une carte graphique prenant en charge le modèle de programmation CUDA.

Nous allons utiliser une version CSV de ces données, où les colonnes 0..3 contiennent les différentes caractéristiques de l’espèce d'Iris et la colonne 4 contient la classe de l’enregistrement, ou l’espèce d'Iris, codée avec la valeur 0, 1 ou 2:


##Vectoriser et lire les données
Nous codons la classe avec un nombre car les réseaux de neurones fonctionnent avec des nombres. **La transformation d’éléments de données du monde réel en séries de nombres (vecteurs) est appelée vectorisation** - deeplearning4j utilise la bibliothèque datavec pour le faire.


Commençons par utiliser cette bibliothèque pour entrer le fichier avec les données vectorisées. Lors de la création de _CSVRecordReader_ , nous pouvons spécifier le nombre de lignes à ignorer (par exemple, si le fichier a une ligne d’en-tête) et le symbole de séparation (dans notre cas, une virgule):


**BatchSize** = permet de décuper le dataSet en segment suivant une taille. Chaque segment du dataset est ensuite transmis pour etre analyser et apporter d'éventuelle ajustement avant de lui fournir le suivant
index = 4 represente le model
nombreEpoche : demande de rejouer le record set


``````shell script
{
  "backpropType" : "Standard",
  "cacheMode" : "NONE",
  "confs" : [ {
    "cacheMode" : "NONE",
    "dataType" : "FLOAT",
    "epochCount" : 0,
    "iterationCount" : 0,
    "layer" : {
      "@class" : "org.deeplearning4j.nn.conf.layers.DenseLayer",
      "activationFn" : {
        "@class" : "org.nd4j.linalg.activations.impl.ActivationSigmoid"
      },
      "biasInit" : 0.0,
      "biasUpdater" : null,
      "constraints" : null,
      "gainInit" : 1.0,
      "gradientNormalization" : "None",
      "gradientNormalizationThreshold" : 1.0,
      "hasBias" : true,
      "hasLayerNorm" : false,
      "idropout" : null,
      "iupdater" : {
        "@class" : "org.nd4j.linalg.learning.config.Adam",
        "beta1" : 0.9,
        "beta2" : 0.999,
        "epsilon" : 1.0E-8,
        "learningRate" : 0.001
      },
      "layerName" : "layer0",
      "nin" : 4,
      "nout" : 10,
      "regularization" : [ ],
      "regularizationBias" : [ ],
      "timeDistributedFormat" : null,
      "weightInitFn" : {
        "@class" : "org.deeplearning4j.nn.weights.WeightInitXavier"
      },
      "weightNoise" : null
    },
    "maxNumLineSearchIterations" : 5,
    "miniBatch" : true,
    "minimize" : true,
    "optimizationAlgo" : "STOCHASTIC_GRADIENT_DESCENT",
    "seed" : 1589550805944,
    "stepFunction" : null,
    "variables" : [ "W", "b" ]
  }, {
    "cacheMode" : "NONE",
    "dataType" : "FLOAT",
    "epochCount" : 0,
    "iterationCount" : 0,
    "layer" : {
      "@class" : "org.deeplearning4j.nn.conf.layers.OutputLayer",
      "activationFn" : {
        "@class" : "org.nd4j.linalg.activations.impl.ActivationSoftmax"
      },
      "biasInit" : 0.0,
      "biasUpdater" : null,
      "constraints" : null,
      "gainInit" : 1.0,
      "gradientNormalization" : "None",
      "gradientNormalizationThreshold" : 1.0,
      "hasBias" : true,
      "idropout" : null,
      "iupdater" : {
        "@class" : "org.nd4j.linalg.learning.config.Adam",
        "beta1" : 0.9,
        "beta2" : 0.999,
        "epsilon" : 1.0E-8,
        "learningRate" : 0.001
      },
      "layerName" : "layer1",
      "lossFn" : {
        "@class" : "org.nd4j.linalg.lossfunctions.impl.LossMSLE"
      },
      "nin" : 10,
      "nout" : 3,
      "regularization" : [ ],
      "regularizationBias" : [ ],
      "timeDistributedFormat" : null,
      "weightInitFn" : {
        "@class" : "org.deeplearning4j.nn.weights.WeightInitXavier"
      },
      "weightNoise" : null
    },
    "maxNumLineSearchIterations" : 5,
    "miniBatch" : true,
    "minimize" : true,
    "optimizationAlgo" : "STOCHASTIC_GRADIENT_DESCENT",
    "seed" : 1589550805944,
    "stepFunction" : null,
    "variables" : [ "W", "b" ]
  } ],
  "dataType" : "FLOAT",
  "epochCount" : 0,
  "inferenceWorkspaceMode" : "ENABLED",
  "inputPreProcessors" : { },
  "iterationCount" : 0,
  "tbpttBackLength" : 20,
  "tbpttFwdLength" : 20,
  "trainingWorkspaceMode" : "ENABLED",
  "validateOutputLayerConfig" : true
}

``````



## entrainer et generer un model préentrainer. 

````java
package youcef.dl4J;

import java.io.File;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class IrisApp {

	
	// file Csv columns
	  private static final int COLUMN_SEPALLENGTH = 0 ;
	  private static final int COLUMN_SEPALWIDTH = 1 ;
	  private static final int COLUMN_PETALLENGTH = 2 ;
	  private static final int COLUMN_PETALWIDTH = 3 ;
	  private static final int COLUMN_IRISFLOUR = 4 ;
	  
	  // number iris flours
	  private static final int IRISFLOURCOUNT = 3 ;

	  
	public static void main(String[] args) throws Exception {

		Double learningRate = 0.001;
		int numInputs = 4;   // couche d'Iris d'entré ( caractériques d'Iris )
		int numHidden = 10;  // couche de neuronnes
		int numOutputs = IRISFLOURCOUNT;  // couche de sortie ( 3 type Iris )
		
		
		
		
		
		System.out.println("----------------------------------------");
		System.out.println("CREATION DU MODEL");
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder().updater(new Adam(learningRate))

				.list()
				.layer(0,
						new DenseLayer.Builder()
						.weightInit(WeightInit.XAVIER)
						.nIn(numInputs)
						.nOut(numHidden)
						.activation(Activation.SIGMOID)
						.build())
				.layer(1,
						new OutputLayer.Builder()  // Creer une Couche de sortie
						.nIn(numHidden) // couche en entrée de la précédente couche
						.nOut(numOutputs )  // couche de sorite
						.activation(Activation.SOFTMAX)  
						.lossFunction(LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR) // fonction d'erreur quadratique
						.build())
				.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(configuration);
		model.init();
		
		// affiche le model
		System.out.println(configuration.toJson());
		
		
		
		System.out.println("----------------------------------------");
		System.out.println("CREATE UI SERVER");
				
		UIServer uiServer = UIServer.getInstance();
		StatsStorage  inMemoryStatsStorage= new InMemoryStatsStorage();
		uiServer.attach(inMemoryStatsStorage);
		
		model.setListeners(new StatsListener(inMemoryStatsStorage));
			
		
		System.out.println("----------------------------------------");
		System.out.println("ENTRAINEMENT DU MODEL");

		File fileTrain=new ClassPathResource("dataset-iris.csv").getFile();
		
		// dataVec pour la vectorisation des données
		RecordReader recordReaderTrain = new CSVRecordReader();
		
		recordReaderTrain.initialize(new FileSplit(fileTrain));
		
		int batchSize=1; // demande de traiter par lot de ligne.
		int  numPossibleLabel = numOutputs; // nombre de fleur possible

		DataSetIterator dataSetIteratorTrain=new RecordReaderDataSetIterator(recordReaderTrain, batchSize,COLUMN_IRISFLOUR, IRISFLOURCOUNT) ;
		
	
		
		while(dataSetIteratorTrain.hasNext()) {  // interate batchSize
			DataSet dataSetTrain = dataSetIteratorTrain.next();
			System.out.println("--------------------------------------------------");
			System.out.println("getFeatures: " + dataSetTrain.getFeatures());
			System.out.println("getLabels: " + dataSetTrain.getLabels());
			System.out.println("getColumnNames:" + dataSetTrain.getColumnNames());
			System.out.println("getExampleMetaData:" + dataSetTrain.getExampleMetaData());
			System.out.println("getLabelNamesList:" + dataSetTrain.getLabelNamesList());			
		}
		
	
		
		// demande de rejouer 100 fois le batchsize
		int numberEpoch = 10;
		for (int i = 0; i < numberEpoch ; i++) {
			model.fit(dataSetIteratorTrain);
		}
		 
		
		
		System.out.println("----------------------------------------");
		System.out.println("VALIDATION DU MODEL PAR SERIE TEST");

		File fileTest=new ClassPathResource("test-iris.csv").getFile();
		
		// dataVec pour la vectorisation des données
		RecordReader recordReaderTest = new CSVRecordReader();
		
		recordReaderTest.initialize(new FileSplit(fileTest));

		int batchSizeTest=1;
		DataSetIterator dataSetIteratorTest=new RecordReaderDataSetIterator(recordReaderTest, batchSizeTest,COLUMN_IRISFLOUR, IRISFLOURCOUNT) ;

	    Evaluation evaluation = new Evaluation();
	    while(dataSetIteratorTest.hasNext()) {
			DataSet dataSetTest = dataSetIteratorTest.next();
		//	System.out.println("--------------------------------------------------");
		//	System.out.println("getFeatures: " + dataSetTest.getFeatures());
		//	System.out.println("getLabels: " + dataSetTest.getLabels());
		//	System.out.println("getColumnNames:" + dataSetTest.getColumnNames());
		//	System.out.println("getExampleMetaData:" + dataSetTest.getExampleMetaData());
			
			INDArray features = dataSetTest.getFeatures();
			INDArray Targetlabels =  dataSetTest.getLabels();
			INDArray predictedLabel = model.output(features);
			evaluation.eval(predictedLabel, Targetlabels);
			
			System.out.println(evaluation.stats());
	    }
		
		
	    boolean saveUpdater=true; // fait reference à updater. Permet la mise par d'autre model externe.
		
	    // genere un mdoelPredefini
	    ModelSerializer.writeModel(model, "irisModel.zip", saveUpdater);
	
		
		/*
		System.out.println("----------------------------------------");
		System.out.println("PREDICTION DE DONNEES");
	    
		INDArray inputPrediction = Nd4j.create(new double[][] {
			{5.1,3.5,1.4,0.2},
			{4.9,3.0,1.4,0.2},
			{6.7,3.1,4.4,1.4},
			{5.6,3.0,4.5,1.5},
			{6.0,3.0,4.8,1.8},
			{6.9,3.1,5.4,2.1}
		});
		
		
		INDArray outputPrediction = model.output(inputPrediction);
		
		System.out.println(outputPrediction);

	
			
		// recuperer l'index du max. exemple 0 =>  [[    0.9693,    0.0306, 7.7953e-5],
		int RECUPERE_PAR_LINE = 1;
		int RECUPERE_PAR_COLUMN = 2;

	    // PredictionIndexFlour: return Iris flour probability.
		int[] predictionIndexFlour=outputPrediction.argMax(RECUPERE_PAR_LINE).toIntVector();
		
		 for (int i = 0; i < predictionIndexFlour.length; i++) {
			System.out.println("flour: " + labelsIrisFlours[predictionIndexFlour[i]]);
		}
					
		
		 Scanner scanner = new Scanner( System.in );
		 System.out.print( "Veuillez saisir key : " );
         int a = scanner.nextInt();
         */
		
		
	}

}

````


## On peut visualiser le réseau va interface deeplearing4j ui

````shell script
http:localhost:9000
````

## Test le model préentratiner
````java
package youcef.dl4J;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class predictionIrisFlour {

	public static void main(String[] args) throws Exception {

		String[] labelsIrisFlours = new String[] { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

		System.out.println("----------------------------------------");
		System.out.println("PREDICTION DE DONNEES");

		// charge notre model pré-entrainer dans IrisApp.java 
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));

		// demande la prédiction de 6 fleurs avec leur "features"
		INDArray inputPrediction = Nd4j.create(new double[][] { 
			{ 5.1, 3.5, 1.4, 0.2 }, 
			{ 4.9, 3.0, 1.4, 0.2 },
				{ 6.7, 3.1, 4.4, 1.4 }, 
				{ 5.6, 3.0, 4.5, 1.5 }, 
				{ 6.0, 3.0, 4.8, 1.8 }, 
				{ 6.9, 3.1, 5.4, 2.1 } });

		// renvoi la prediction
		INDArray outputPrediction = model.output(inputPrediction);

		System.out.println(outputPrediction);

		// recuperer l'index du max. exemple 0 => [[ 0.9693, 0.0306, 7.7953e-5],
		int RECUPERE_PAR_LINE = 1;
		int RECUPERE_PAR_COLUMN = 2;

		// PredictionIndexFlour: return Iris flour probability.
		int[] predictionIndexFlour = outputPrediction.argMax(RECUPERE_PAR_LINE).toIntVector();

		for (int i = 0; i < predictionIndexFlour.length; i++) {
			System.out.println("flour: " + labelsIrisFlours[predictionIndexFlour[i]]);
		}

		Scanner scanner = new Scanner(System.in);
		System.out.print("Veuillez saisir key : ");
		int a = scanner.nextInt();

	}

}

````


