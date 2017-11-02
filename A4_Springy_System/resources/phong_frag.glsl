#version 120

uniform sampler2D texture_cloth_1;

varying vec2 vTex;
uniform vec3 kdFront;
uniform vec3 kdBack;

varying vec3 vPos; // in camera space
varying vec3 vNor; // in camera space

void main()
{
	
	vec3 lightPos = vec3(0.0, 0.0, 0.0);
	vec3 n = normalize(vNor);
	vec3 l = normalize(lightPos - vPos);
	vec3 v = -normalize(vPos);
	vec3 h = normalize(l + v);
	vec3 kd = kdFront;
	float ln = dot(l, n);
	if(ln < 0.0) {
		kd = kdBack;
		ln = -ln;
	}
	vec3 diffuse = ln * kd;
	vec3 color = diffuse;

	vec3 tt = texture2D(texture_cloth_1, vTex).rgb;
	
	gl_FragColor = vec4(0.8*tt + 0.2*color, 1.0);


}