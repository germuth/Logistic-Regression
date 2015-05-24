package ca.germuth.machine_learning;

import Jama.Matrix;

public class GradientDescent {

	//if you start getting NaN, then the error is increasing every time, because your learning rate is way to high
	//TODO detect when error has increased since last iteration
	public static Matrix run(Matrix X, Matrix y, final int NUM_ITERATIONS,
			double LEARNING_RATE, double LAMBDA, boolean REGULARIZED, boolean PRINT_PROGRESS) {
	//n = number of features (including fake feature)
		int n = X.getColumnDimension();
		//m = number of training tuples
		int m = X.getRowDimension();
		
		// initialize theta to be all ones, just need a starting state
		double[][] thetaArr = new double[n][1];
		for (int i = 0; i < thetaArr.length; i++) {
			thetaArr[i][0] = 0.0;
		}
		Matrix theta = new Matrix(thetaArr);

		double currError = Main.error(theta, X, y);
		if(PRINT_PROGRESS){
			System.out.println("Error before: " + currError);
		}
		// difficult to test for convergence, so rather
		// just do 100 fixed iterations
		for (int i = 0; i < NUM_ITERATIONS; i++) {
			// where x(0) = 1.0

			double[][] newTheta = new double[n][1];
			// for each feature, calc partial derivative
			// dJ/d(theta) = 1/m * sum( predicted - actual ) * x(i)
			for (int j = 0; j < n; j++) {
				double currTheta = theta.getArray()[j][0];
				double deriv = 0.0;
				for (int row = 0; row < m; row++) {
					deriv += (Main.predict(theta, X.getMatrix(row, row, 0, n-1)) - y.getArray()[row][0]) * X.getArray()[row][j];
				}
				//error function changes with regularization, so the derivative does as well
				if(REGULARIZED){
					deriv += LAMBDA * currTheta;
				}
				newTheta[j][0] = currTheta - LEARNING_RATE * deriv / m;
			}

			theta = new Matrix(newTheta);

			currError = Main.error(theta, X, y);
			if(PRINT_PROGRESS){
				System.out.println(currError);				
			}
		}
		return theta;
	}
}
