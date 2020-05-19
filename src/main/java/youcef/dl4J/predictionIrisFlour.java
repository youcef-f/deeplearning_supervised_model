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
