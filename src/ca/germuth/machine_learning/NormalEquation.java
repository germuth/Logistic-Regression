package ca.germuth.machine_learning;

import Jama.Matrix;

public class NormalEquation {
	public static Matrix run(Matrix X, Matrix y){
		//theta = inv(X' * X) * X' * y
		Matrix op1 = X.transpose();
		Matrix op2 = (op1.times(X)).inverse();
		Matrix op3 = op2.times(op1).times(y);
		return op3;
	}
}
