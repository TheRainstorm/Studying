package test;

public class RoundFunc {
	//xΪ����С����nΪ�������뵽С�����ڼ�λС����0�����������뵽����
	static double round(double x,int n) {
		if(n==0) {
			return Math.floor(x+0.5);
			//return Math.round(x);
		}
		else {
			long r=(long)Math.pow(10,n);
			double x2=x*r;
			x2=Math.floor(x2+0.5);
			return x2/r;
		}
	}
	public static void main(String[] args) {
		double x=1.455;
		System.out.println(""+x+"="+round(x,0)+"");
	}

}
