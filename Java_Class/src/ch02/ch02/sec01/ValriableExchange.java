package ch02.ch02.sec01;

/**
 * ch02 예제: x, y 초기값 출력 및 임시변수를 이용한 값 교환
 */
public class ValriableExchange {

	public static void main(String[] args) {
		// 1. x, y에 초기값 지정 후 출력
		int x = 10;
		int y = 20;

		System.out.println("--- 교환 전 ---");
		System.out.println("x = " + x);
		System.out.println("y = " + y);

		// 2. 임시변수를 사용하여 x와 y의 값 교환
		int temp = x;
		x = y;
		y = temp;

		System.out.println("--- 교환 후 ---");
		System.out.println("x = " + x);
		System.out.println("y = " + y);
	}
}
