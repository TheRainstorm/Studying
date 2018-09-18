package enigma;

public class EnigmaTest {

	public static void main(String[] args) {
		Enigma enigma=new Enigma(4,6,1);
		String str="threeblueonebrown";
		String str2=enigma.encrypt(str);
		System.out.println(str2);
		Enigma enigma2=new Enigma(4,6,1);
		String str3=enigma2.decrypt(str2);
		System.out.println(str3);
		
//		char ch=enigma.encrypt('h');
//		System.out.println(ch);
	}

}
