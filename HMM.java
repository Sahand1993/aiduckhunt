import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.Math;

public class MainHmm3 {

    public static BufferedReader br;
    public static double[][] A;
    public static double[][] B;
    public static double[][] pi;
    public static int[] O;
    public static double[][] alpha;
    public static double[][] beta; // i, t
    public static double[][][] digamma; // i, j, t
    public static double[][] gamma; // i, t
    public static int maxIters = 1000;
    public static double[] colSums;
    public static double logProb;
    static double[][] delta;
    static int[][] deltaIndex;

    public static void main(String[] args) {
        br = new BufferedReader(new InputStreamReader(System.in));
        readInput();

        fit();

        System.out.printf("%d %d ", A.length, A[0].length);
        for (int i = 0; i < A.length; i++) {
            for(int j = 0; j < A[0].length; j++) {
                System.out.printf("%f ", A[i][j]);
            }
        }
        System.out.println();
        System.out.printf("%d %d ", B.length, B[0].length);
        for (int i = 0; i < B.length; i++){
            for(int j = 0; j < B[0].length; j++) {
                System.out.printf("%f ", B[i][j]);
            }
        }
    }

    public static double fit(){
        estimateParams();
        double newLogProb = computeLogP();
        int iters = 0;
        do {
            //System.out.printf("logProb: %f\n", newLogProb);
            logProb = newLogProb;
            estimateParams();
            newLogProb = computeLogP();
            iters++;
        } while(iters < maxIters && (logProb - newLogProb) < -0.00001);
        System.out.printf("Stopped after %d iterations.\n", iters);
        return newLogProb;
    }

    public static double computeLogP() {
        double logProb = 0;
        for(int i = 0; i < O.length; i++) {
            logProb += Math.log(1 / colSums[i]); // Does it matter which log-base is used?
        }
        logProb = -logProb;
        return logProb;
    }

    public static void estimateParams() {
        fillAlpha();
        fillBeta();
        fillGammas();

        for(int i = 0; i < A.length; i++) {
            pi[0][i] = gamma[i][0];
        }

        double numer;
        double denom;
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < A.length; j++) {
                numer = 0;
                denom = 0;
                for(int t = 0; t < O.length - 1; t++) {
                    numer += digamma[i][j][t];
                    denom += gamma[i][t];
                }
                A[i][j] = numer / denom;
            }
        }

        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < B[0].length; j++) {
                numer = 0;
                denom = 0;
                for(int t = 0; t < O.length; t++) {
                    if(O[t] == j) {
                        numer += gamma[i][t];
                    }
                    denom += gamma[i][t];
                }
                B[i][j] = numer / denom;
            }
        }
    }

    public static void fillAlpha() {
        alpha = new double[A.length][O.length];
        colSums = new double[O.length];
        // First column of alpha
        double colSum = 0;
        for (int i = 0; i < alpha.length; i++) {
            alpha[i][0] = pi[0][i] * B[i][O[0]];
            colSum += alpha[i][0];
        }
        colSums[0] = colSum;
        normalizeCol(alpha, 0, colSum);

        double[][] alphaOld;
        double[][] newColAlpha = extractColumn(alpha, 0);
        for (int i = 1; i < O.length; i++) {
            alphaOld = extractColumn(alpha, i - 1);
            newColAlpha = matrixMul(transpose(alphaOld), A);
            newColAlpha = vectorMul(transpose(newColAlpha), extractColumn(B, O[i]));
            colSum = 0;
            for(int j = 0; j < alpha.length; j++) {
                alpha[j][i] = newColAlpha[j][0];
                colSum += alpha[j][i];
            }
            colSums[i] = colSum;
            normalizeCol(alpha, i, colSum);
        }
    }

    private static void fillBeta() {
        beta = new double[A.length][O.length];
        // Last col of beta
        for (int i = 0; i < beta.length; i++) {
            beta[i][beta[0].length - 1] = 1 / colSums[O.length - 1];
        }
        for (int t = O.length - 2; t >= 0; t--) {
            for (int i = 0; i < A.length; i++) {
                double sum = 0;
                for (int j = 0; j < A.length; j++) {
                    sum += beta[j][t + 1] * B[j][O[t + 1]] * A[i][j];
                }
                beta[i][t] = sum / colSums[t];
            }
        }
    }

    /**
     * Fills delta and deltaIndex matrices with probabilities.
     */
    public static void fillDelta(){
        delta = new double[A.length][O.length];
        deltaIndex = new int[A.length][O.length];

        firstColDelta();

        double[] probs;
        // For each timestep t
        for(int t = 1; t < O.length; t++) {
            // For each possible state at t
            for(int i = 0; i < A.length; i++) {
                probs = new double[A.length];
                // For each possible state at t - 1
                for(int j = 0; j < A.length; j++) {
                    probs[j] = delta[j][t - 1] * A[j][i] * B[i][O[t]];
                }
                DoubleInt doubleInt = max(probs);
                delta[i][t] = doubleInt.getMax();
                deltaIndex[i][t] = doubleInt.getArgmax();
            }
        }
    }


    /**
     *  Fills the first column of delta
     */
    public static void firstColDelta(){
        for(int i = 0; i < A.length; i++) {
            delta[i][0] = pi[0][i] * B[i][O[0]];
        }
    }

    /**
     * Return max of array
     */
    public static DoubleInt max(double[] arr){
        double max = arr[0];
        int argmax = 0;
        for(int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                argmax = i;
            }
        }
        return new DoubleInt(max, argmax);
    }

    /**
     * Finds the most likely path by backtracking in deltaIndex.
     */
    public static String findSequence(){
        String seq = "";
        double maxProb = 0;
        int iMax = -1;
        for (int i = 0; i < A.length; i++) {
            double tempProb = delta[i][delta[0].length - 1];
            if (tempProb > maxProb) {
                maxProb = tempProb;
                iMax = i;
            }
        }
        if (maxProb > 0) {
            seq = backTrack(iMax, delta[0].length - 1);
        }
        return seq;
    }

    /**
     *  Returns the sequence of states from backtracking as a String.
     * @param i is the state we're backtracking from.
     * @param t is the timestep where we have state i.
     * @return the entire backtracking sequence
     */
    public static String backTrack(int i, int t){
        if (t == 0) {
            return "" + i + " ";
        }
        return backTrack(deltaIndex[i][t], t - 1) + i + " ";
    }

    private static class DoubleInt {
        private double max;
        private int argmax;
        public DoubleInt(double max, int argmax) {
            this.max = max;
            this.argmax = argmax;
        }

        public double getMax() {
            return max;
        }

        public void setMax(double max) {
            this.max = max;
        }

        public int getArgmax() {
            return argmax;
        }

        public void setArgmax(int argmax) {
            this.argmax = argmax;
        }
    }

    private static void fillGammas() {
        gamma = new double[A.length][O.length];
        digamma = new double[A.length][A.length][O.length - 1];
        for (int t = 0; t < O.length - 1; t++) {
            for (int i = 0; i < A.length; i++) {
                gamma[i][t] = 0;
                for (int j = 0; j < A.length; j++) {
                    digamma[i][j][t] = alpha[i][t] * A[i][j] * B[j][O[t + 1]] * beta[j][t + 1];
                    gamma[i][t] += digamma[i][j][t];
                }
            }
        }
        for (int i = 0; i < A.length; i++) {
            gamma[i][O.length - 1] = alpha[i][O.length - 1];
        }
    }

    public static void printMatrix(double[][][] m) {
        for(int t = 0; t < m[0][0].length; t++) {
            for(int i = 0; i < m.length; i++) {
                for(int j = 0; j < m[0].length; j++){
                    System.out.printf("%f ", m[i][j][t]);
                }
                System.out.println();
            }
            System.out.println();
        }
        System.out.println();
    }

    private static void normalizeCol(double[][] m, int colNo, double colSum) {
        for (int i = 0; i < m.length; i++) {
            m[i][colNo] /= colSum;
        }
    }

    private static void printCol(double[][] m, int colNo){
        for (int i = 0; i < m.length; i++) {
            System.out.printf("%f\n", m[i][colNo]);
        }
    }


    public static double[][] vectorMul(double[][] a, double[][] b){
        double[][] res = new double[a.length][1];
        for(int i = 0; i < a.length; i++) {
            res[i][0] = a[i][0] * b[i][0];
        }
        return res;
    }

    public static double[][] extractColumn(double[][] m, int colNo){
        double[][] res = new double[m.length][1];
        for(int i = 0; i < m.length; i++) {
            res[i][0] = m[i][colNo];
        }
        return res;
    }

    public static double[][] transpose(double[][] trans) {
        double[][] res = new double[trans[0].length][trans.length];
        for (int i = 0; i < trans.length; i++) {
            for (int j = 0; j < trans[0].length; j++) {
                res[j][i] = trans[i][j];
            }
        }
        return res;
    }
    public static double sumMatrix(double[][] matrix) {
        double sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            sum += matrix[i][0];
        }
        return sum;
    }
    public static void printMatrix(double[][] m){
        for(int i = 0; i < m.length; i++) {
            for( int j = 0; j < m[0].length; j++) {
                System.out.printf("%2f ", m[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void printMatrix(int[][] m) {
        for(int i = 0; i < m.length; i++) {
            for( int j = 0; j < m[0].length; j++) {
                System.out.print(m[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void printVector(int[] v) {
        for(int i = 0; i < v.length; i++) {
            System.out.print(v[i] + " ");
        }
        System.out.println();
    }
}
