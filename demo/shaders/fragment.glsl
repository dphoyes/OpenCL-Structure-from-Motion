#version 300 es

in lowp float distance;
in lowp float reflectance;
in lowp float argh;
out lowp vec3 frag_colour;

uniform lowp vec3 colour;
 
void main()
{
    const lowp vec3 fog_colour = vec3(0.1, 0.1, 0.1);

    lowp float fogginess = pow(distance / 200.0, 2);

//    frag_colour = mix(colour, fog_colour, clamp(fogginess, 0, 1));
//    frag_colour = colour + vec3(1.0)*pow(distance/10.0, -2);

    if (colour == vec3(1,1,1))
    {
        frag_colour = colour;
    }
    else
    {
        frag_colour = colour + vec3(0.6)*reflectance;
    }
}
