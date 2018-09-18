package sort;

public class Sort {

	public static void main(String[] args) {
		char[] chs1=new String("abcdefghijklmnopqrstuvwxyz").toCharArray();
		char[] chs2=new String("yruhqsldpxngokmiebfzcwvjat").toCharArray();
		for(int i=0;i<25;i++) {
			int min=i;
			for(int j=i+1;j<26;j++) {
				if(chs2[j]<chs2[min]) {
					min=j;
				}
			}
			if(min!=i) {
				char temp1=chs1[i];
				chs1[i]=chs1[min];
				chs1[min]=temp1;
				
				char temp2=chs2[i];
				chs2[i]=chs2[min];
				chs2[min]=temp2;
			}
		}
		
		System.out.println(chs1);
		boolean b='a'>'b';
		System.out.println(b);
	}

}
