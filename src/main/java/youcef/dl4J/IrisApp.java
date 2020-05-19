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
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				.updater(new Adam(learningRate))
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
