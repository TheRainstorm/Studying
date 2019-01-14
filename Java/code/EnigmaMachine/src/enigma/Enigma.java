package enigma;

public class Enigma {
	private static char[] rotor1;
	private static char[] rotor2;
	private static char[] rotor3;
	private static char[] reflector;
	private static char[] rotor12;
	private static char[] rotor22;
	private static char[] rotor32;
	static {
		rotor1=new String("ekmflgdqvzntowyhxuspaibrcj").toCharArray();
		rotor2=new String("ajdksiruxblhwtmcqgznpyfvoe").toCharArray();
		rotor3=new String("bdfhjlcprtxvznyeiwgakmusqo").toCharArray();
	 reflector=new String("yruhqsldpxngokmiebfzcwvjat").toCharArray();
	   rotor12=new String("uwygadfpvzbeckmthxslrinqoj").toCharArray();
	   rotor22=new String("ajpczwrlfbdkotyuqgenhxmivs").toCharArray();
	   rotor32=new String("tagbpcsdqeufvnzhyixjwlrkom").toCharArray();
	}
	private static char[][] array= {rotor3,rotor2,rotor1,reflector,
			rotor12,rotor22,rotor32};
	//三个转子的初始位置
	//0代表A,1代表B,...,25代表Z
	private int A,B,C;
	
	public Enigma() {
		A=0;B=0;C=0;
	}
	public Enigma(int a,int b,int c) {
		assert 0<=a && a<26 && 0<=b && b<26 && 0<=c && c<26;
		A=a;B=b;C=c;
	}
	public void setABC(int a,int b,int c) {
		assert 0<=a && a<26 && 0<=b && b<26 && 0<=c && c<26;
		A=a;B=b;C=c;
	}
	
	public char encrypt(char ch) {
		ch=lowercase(ch);
		int n=ch-'a';//要加密的字母对应的数字（0~25）
		
		int[] list= {0,C,B,A,0,A,B,C,0};//存储转子的位置
		for(int i=0;i<7;i++) {
			n=transfer(n,list[i],list[i+1]);//外部对应
			n=array[i][n]-'a';//转子内部对应
		}
		n=transfer(n,C,0);//最后一次外部对应
		
		//不用二维数组的方法
//		n=transfer(n,0,C);
//		n=rotor3[n]-'a';
//		n=transfer(n,C,B);
//		n=rotor2[n]-'a';
//		n=transfer(n,B,A);
//		n=rotor1[n]-'a';
//		n=transfer(n,A,0);
//		n=reflector[n]-'a';
//		n=transfer(n,0,A);
//		n=rotor12[n]-'a';
//		n=transfer(n,A,B);
//		n=rotor22[n]-'a';
//		n=transfer(n,B,C);
//		n=rotor32[n]-'a';
//		n=transfer(n,C,0);
		
		addABC();//没加密一个字母转子转动
		return (char)(n+'a');
	}
	public String encrypt(String str) {
		char[] chArray=str.toCharArray();
		String result="";
		for(int i=0;i<chArray.length;i++) {
			result+=encrypt(chArray[i]);
		}
		return result;
	}
	public String decrypt(String str) {
		return encrypt(str);
	}
	//每加密一个字符，转子转动一格
	private void addABC() {
		if(C+1==26) {
			C=0;
			if(B+1==26) {
				B=0;
				A=(A+1)%26;
			}
			else {
				B++;
			}
		}
		else {
			C++;
		}
	}
	//两个转子位置分别为n1,n2,返回ch从第一个到第二对应的字符
	private int transfer(int ch,int n1,int n2) {
		return (ch+n2-n1+26)%26;
	}
	//大小写字母统一返回小写字母
	private char lowercase(char ch) {
		return (0<=ch-'a' && ch-'a'<26)?ch:(char)(ch+32);
	}
}
