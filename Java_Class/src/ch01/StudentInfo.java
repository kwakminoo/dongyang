package ch01;

import java.io.PrintStream;
import java.util.Scanner;
import java.nio.charset.StandardCharsets;

/**
 * ch01 예제: 이름, 나이, 성별, 학번을 입력받아 출력 (한글 UTF-8 지원)
 */
public class StudentInfo {

	public static void main(String[] args) {
		// 한글 출력이 깨지지 않도록 UTF-8로 설정
		try {
			System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
		} catch (Exception e) {
			// 설정 실패 시 기본 출력 사용
		}

		Scanner sc = new Scanner(System.in, StandardCharsets.UTF_8);

		System.out.print("이름: ");
		String name = sc.nextLine();

		System.out.print("나이: ");
		int age = sc.nextInt();
		sc.nextLine(); // 숫자 입력 후 남은 줄바꿈 제거

		System.out.print("성별: ");
		String gender = sc.nextLine();

		System.out.print("학번: ");
		String studentId = sc.nextLine();

		System.out.println("--- 입력 정보 ---");
		System.out.println("이름: " + name);
		System.out.println("나이: " + age);
		System.out.println("성별: " + gender);
		System.out.println("학번: " + studentId);

		sc.close();
	}
}
