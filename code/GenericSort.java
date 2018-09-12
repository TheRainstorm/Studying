package sort;

public class GenericSort {
	public static void sort(Object[] objectArray) {
		Object temp;
		for(int i=0;i<objectArray.length-1;i++) {
			int minIndex=i;
			for(int j=i+1;j<objectArray.length;j++) {
				if(((Comparable)objectArray[minIndex]).compareTo(objectArray[j])>0) {
					minIndex=j;
				}
			}
			if(minIndex!=i) {
				temp=objectArray[i];
				objectArray[i]=objectArray[minIndex];
				objectArray[minIndex]=temp;
			}
		}
	}
	
	public static void print(Object[] objectArray) {
		for(int i=0;i<objectArray.length;i++) {
			System.out.print(objectArray[i]+" ");
		}
		System.out.println();
	}

	public static void main(String[] args) {
		Integer[] intArray= {4,3,1,2};
		Double[] doubleArray= {1.4,1.2,1.3,1.1};
		String[] strArray= {"dog","cat","big","about"};
		
		print(intArray);
		print(doubleArray);
		print(strArray);
		
		sort(intArray);
		sort(doubleArray);
		sort(strArray);
		
		print(intArray);
		print(doubleArray);
		print(strArray);
		

	}

}
